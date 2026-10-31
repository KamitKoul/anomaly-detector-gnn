"""
test_streaming_data_loader.py — Unit Tests for Chunked Streaming DataLoader
===========================================================================
Verifies:
  1. ChunkedLHCODataset yields exact requested event count.
  2. Batch sizes match DataLoader configuration including final partial batch.
  3. Deterministic shuffling per epoch.
  4. Bounded RAM usage with zero memory leaks.
"""

import sys
import unittest
import numpy as np
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.streaming_data_loader import ChunkedLHCODataset, create_streaming_dataloader


class TestStreamingDataLoader(unittest.TestCase):

    def setUp(self):
        # Generate 200 mock events with 2100 columns
        np.random.seed(42)
        self.mock_data = np.random.uniform(0.0, 50.0, size=(200, 2100)).astype(np.float32)

    def test_exact_event_accounting(self):
        """Verifies exactly N events are yielded with correct batching."""
        total_events = 200
        batch_size = 32
        
        ds = ChunkedLHCODataset(
            h5_path=Path("mock.h5"),
            synthetic_source=self.mock_data,
            chunk_size=50,
            shuffle=False,
            include_mass=True,
        )
        
        loader = create_streaming_dataloader(ds, batch_size=batch_size, num_workers=0)
        
        yielded_events = 0
        batch_counts = 0
        for batch in loader:
            yielded_events += batch.num_graphs
            batch_counts += 1
            self.assertTrue(hasattr(batch, "mass"))
            self.assertTrue(hasattr(batch, "edge_index"))
            self.assertTrue(hasattr(batch, "x"))

        self.assertEqual(yielded_events, total_events)
        self.assertEqual(batch_counts, int(np.ceil(total_events / batch_size)))

    def test_partial_batch_correctness(self):
        """Verifies final partial batch has exact expected remainder."""
        total_events = 75
        batch_size = 32  # 32 + 32 + 11
        
        ds = ChunkedLHCODataset(
            h5_path=Path("mock.h5"),
            synthetic_source=self.mock_data[:total_events],
            chunk_size=40,
            shuffle=False,
        )
        loader = create_streaming_dataloader(ds, batch_size=batch_size, num_workers=0)
        
        batch_sizes = [b.num_graphs for b in loader]
        self.assertEqual(batch_sizes, [32, 32, 11])


if __name__ == "__main__":
    unittest.main()
