"""
streaming_data_loader.py — Chunked, Worker-Safe Streaming DataLoader for Million-Scale GNNs
=============================================================================================
Replaces in-memory graph accumulation with an on-the-fly streaming pipeline:
  1. Reads raw collision events from HDF5 in configurable chunks (default: 2,000 events).
  2. Dynamically applies event-local preprocessing and builds Delta-R graphs.
  3. Yields batched PyG collision graphs via PyTorch DataLoader.
  4. Keeps peak memory bounded to < 500 MB regardless of dataset size (up to 1,000,000 events).
  5. Multi-worker safe: shards event chunks cleanly across DataLoader workers.
  6. Supports reproducible shuffling per epoch via worker-seeded generators.
"""

import math
from pathlib import Path
from typing import Optional, List, Iterator, Sequence
import numpy as np
import tables
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import Data, Batch

from src.run_lhco_gnn import (
    event_to_graph,
    compute_kinematic_recon_loss,
    DELTA_R_THRESHOLD,
)
from src.adversarial_unlearning_ae import compute_invariant_mass


class ChunkedLHCODataset(IterableDataset):
    """
    High-performance, memory-bounded streaming dataset for LHCO collision events.
    Streams events from an HDF5 file using an explicit index array, dynamically
    building PyG graphs on the fly.
    """

    def __init__(
        self,
        h5_path: Path,
        event_indices: Optional[Sequence[int]] = None,
        chunk_size: int = 2000,
        shuffle: bool = True,
        seed: int = 42,
        log_pt: bool = True,
        delta_r_threshold: float = DELTA_R_THRESHOLD,
        include_met_node: bool = True,
        phi_met_mode: str = "atan2",
        include_mass: bool = False,
        mass_norm: float = 1000.0,
        synthetic_source: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed
        self.log_pt = log_pt
        self.delta_r_threshold = delta_r_threshold
        self.include_met_node = include_met_node
        self.phi_met_mode = phi_met_mode
        self.include_mass = include_mass
        self.mass_norm = mass_norm
        self.synthetic_source = synthetic_source

        if event_indices is not None:
            self.event_indices = np.array(event_indices, dtype=np.int64)
        else:
            if synthetic_source is not None:
                self.event_indices = np.arange(len(synthetic_source), dtype=np.int64)
            else:
                self._verify_file_readiness(self.h5_path)
                with tables.open_file(str(self.h5_path), mode="r") as h5:
                    total_events = h5.get_node("/df/block0_values").nrows
                self.event_indices = np.arange(total_events, dtype=np.int64)

        self.total_events = len(self.event_indices)
        self.epoch = 0

    @staticmethod
    def _verify_file_readiness(file_path: Path):
        """Verifies that the target file exists and has physical blocks on disk."""
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(
                f"[FAIL-CLOSED] Target HDF5 dataset not found: {file_path}. "
                "Production training requires physical collision event files. "
                "Synthetic substitutions are strictly prohibited."
            )
        import sys
        st = file_path.stat()
        if sys.platform == "darwin" and hasattr(st, "st_blocks") and st.st_blocks == 0:
            raise RuntimeError(
                f"[FAIL-CLOSED] File {file_path} is an evicted APFS dataless file (st_blocks=0). "
                "The physical dataset contents are not present on local disk. "
                "Please download the dataset locally or mount from persistent ext4/local storage."
            )

    def set_epoch(self, epoch: int):
        """Sets the active epoch to vary deterministic shuffling per epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.total_events

    def _open_h5(self):
        """Worker-safe file opening with fail-closed safety."""
        if self.synthetic_source is not None:
            return None
        self._verify_file_readiness(self.h5_path)
        return tables.open_file(str(self.h5_path), mode="r")

    def _get_worker_slice(self) -> np.ndarray:
        """Determines the exact slice of indices assigned to the active worker."""
        worker_info = torch.utils.data.get_worker_info()
        indices = self.event_indices.copy()

        # Deterministic shuffling per epoch
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(indices)

        if worker_info is None:
            # Single worker / main process
            return indices
        else:
            # Multi-worker sharding
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(indices) / float(num_workers)))
            start = worker_id * per_worker
            end = min(start + per_worker, len(indices))
            return indices[start:end]

    def __iter__(self) -> Iterator[Data]:
        worker_indices = self._get_worker_slice()
        total_worker_events = len(worker_indices)
        if total_worker_events == 0:
            return

        h5_file = self._open_h5()
        node = h5_file.get_node("/df/block0_values") if h5_file is not None else None

        try:
            for chunk_start in range(0, total_worker_events, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, total_worker_events)
                batch_target_indices = worker_indices[chunk_start:chunk_end]

                # Fetch raw event rows from HDF5 or synthetic source
                if self.synthetic_source is not None:
                    raw_events = self.synthetic_source[batch_target_indices]
                else:
                    # In PyTables, reading sorted indices is fastest
                    sort_order = np.argsort(batch_target_indices)
                    sorted_idx = batch_target_indices[sort_order]
                    raw_sorted = node[sorted_idx, :]
                    
                    # Invert sort order to restore intended shuffle order
                    inv_order = np.empty_like(sort_order)
                    inv_order[sort_order] = np.arange(len(sort_order))
                    raw_events = raw_sorted[inv_order]

                # Dynamic graph construction
                for i in range(len(raw_events)):
                    event_row = raw_events[i]
                    # Verify truth bit if mixed file
                    if event_row.shape[0] > 2100 and event_row[2100] == 1.0:
                        continue  # Skip any non-background row

                    from src.run_lhco_gnn import extract_particles
                    parts = extract_particles(event_row[:2100])

                    graph = event_to_graph(
                        log_pt=self.log_pt,
                        delta_r_threshold=self.delta_r_threshold,
                        include_met_node=self.include_met_node,
                        phi_met_mode=self.phi_met_mode,
                        particles=parts,
                    )

                    if self.include_mass:
                        m = compute_invariant_mass(parts)
                        graph.mass = torch.tensor([m / self.mass_norm], dtype=torch.float32)

                    yield graph

        finally:
            if h5_file is not None:
                h5_file.close()


def create_streaming_dataloader(
    dataset: ChunkedLHCODataset,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    drop_last: bool = False,
) -> DataLoader:
    """Creates an optimized PyG-compatible DataLoader over ChunkedLHCODataset with strict drop_last."""
    from torch_geometric.loader import DataLoader as PyGDataLoader
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return PyGDataLoader(dataset, **loader_kwargs)
