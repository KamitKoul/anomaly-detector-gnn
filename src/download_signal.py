"""
download_signal.py — Extractor Utility

Downloads the W' signal dataset from the LHC Olympics 2020
R&D dataset hosted on Zenodo (record 4536377).

The R&D dataset contains 1M QCD background + 100k W'→XY signal
events shuffled together with a truth bit appended per event.
This script downloads the full dataset, extracts the signal
events (truth == 1), and saves them to data/W_prime_signal.h5.
"""

import os
import sys
import requests
import numpy as np
import pandas as pd
import tables

from tqdm import tqdm


########################################
# CONFIG
########################################

ZENODO_RECORD_ID = "4536377"
SIGNAL_FILENAME = "events_anomalydetection.h5"

ZENODO_DOWNLOAD_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    f"/files/{SIGNAL_FILENAME}/content"
)

DATA_DIR = "data"
RAW_DOWNLOAD_PATH = os.path.join(DATA_DIR, SIGNAL_FILENAME)
SIGNAL_OUTPUT_PATH = os.path.join(DATA_DIR, "W_prime_signal.h5")

# Expected shape: 1.1M events × 2101 columns (700 particles × 3 features + truth)
EXPECTED_COLUMNS = 2101
EXPECTED_SIGNAL_EVENTS = 100_000  # approximate


########################################
# DOWNLOAD FROM ZENODO
########################################

def download_from_zenodo(url=ZENODO_DOWNLOAD_URL, output_path=RAW_DOWNLOAD_PATH):
    """
    Downloads the R&D dataset HDF5 file from Zenodo with a tqdm
    progress bar. Streams the download to avoid loading the full
    ~2.6 GB file into memory.

    Args:
        url: Direct download URL for the Zenodo file.
        output_path: Local path to save the downloaded file.

    Returns:
        str: Path to the downloaded file.
    """

    # Create data/ directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"[INFO] File already exists: {output_path}")
        print("[INFO] Skipping download. Delete the file to re-download.")
        return output_path

    print(f"[INFO] Downloading R&D dataset from Zenodo record {ZENODO_RECORD_ID}...")
    print(f"[INFO] URL: {url}")
    print(f"[INFO] Saving to: {output_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading",
            ncols=80,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"[INFO] Download complete: {output_path}")
    return output_path


########################################
# EXTRACT SIGNAL EVENTS
########################################

def extract_signal_events(
    raw_path=RAW_DOWNLOAD_PATH, output_path=SIGNAL_OUTPUT_PATH
):
    """
    Reads the combined R&D dataset and extracts only the signal
    events (truth label == 1). The truth label is the last column
    (index 2100) of each event row.

    Saves the extracted signal events to a separate HDF5 file
    at data/W_prime_signal.h5.

    Args:
        raw_path: Path to the full downloaded R&D dataset.
        output_path: Path to save the extracted signal HDF5.

    Returns:
        int: Number of signal events extracted.
    """

    if os.path.exists(output_path):
        print(f"[INFO] Signal file already exists: {output_path}")
        print("[INFO] Skipping extraction. Delete the file to re-extract.")
        return 0

    print(f"[INFO] Extracting signal events from: {raw_path}")

    # Stream through the HDF5 file to avoid loading everything into RAM
    h5file = tables.open_file(raw_path, mode="r")

    try:
        node = h5file.get_node("/df/block0_values")
        total_rows = node.nrows
        print(f"[INFO] Total events in R&D dataset: {total_rows:,}")

        # Process in chunks to keep memory usage low
        chunk_size = 10_000
        signal_chunks = []
        signal_count = 0

        for start in tqdm(
            range(0, total_rows, chunk_size),
            desc="Scanning for signal",
            ncols=80,
        ):
            stop = min(start + chunk_size, total_rows)
            chunk = node[start:stop]

            # Truth label is the last column (index 2100)
            truth_labels = chunk[:, -1]
            signal_mask = truth_labels == 1

            if np.any(signal_mask):
                signal_events = chunk[signal_mask]
                signal_chunks.append(signal_events)
                signal_count += len(signal_events)

    finally:
        h5file.close()

    if signal_count == 0:
        print("[WARNING] No signal events found in the dataset.")
        return 0

    # Concatenate all signal chunks and save
    print(f"[INFO] Found {signal_count:,} signal events.")
    print(f"[INFO] Saving signal events to: {output_path}")

    all_signal = np.vstack(signal_chunks)

    # Save as HDF5 using pandas for compatibility with stream_lhco_loader
    df_signal = pd.DataFrame(all_signal)
    df_signal.to_hdf(output_path, key="df", mode="w", format="fixed")

    print(f"[INFO] Signal extraction complete: {output_path}")
    return signal_count


########################################
# VERIFY DOWNLOADED FILE
########################################

def verify_signal_file(filepath=SIGNAL_OUTPUT_PATH):
    """
    Verifies the downloaded and extracted signal HDF5 file by
    checking that it is a valid HDF5 file, has the expected
    keys, and contains a reasonable number of rows.

    Args:
        filepath: Path to the signal HDF5 file to verify.

    Returns:
        bool: True if verification passes, False otherwise.
    """

    print(f"\n[INFO] Verifying signal file: {filepath}")

    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return False

    # Check file size
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    try:
        h5file = tables.open_file(filepath, mode="r")
    except Exception as e:
        print(f"[ERROR] Cannot open HDF5 file: {e}")
        return False

    try:
        # Check that expected HDF5 nodes exist
        try:
            node = h5file.get_node("/df/block0_values")
        except tables.NoSuchNodeError:
            print("[ERROR] Missing expected HDF5 node: /df/block0_values")
            return False

        num_rows = node.nrows
        num_cols = node.shape[1] if len(node.shape) > 1 else 0

        print(f"  Rows (events): {num_rows:,}")
        print(f"  Columns: {num_cols}")

        # Validate column count
        if num_cols != EXPECTED_COLUMNS:
            print(
                f"[WARNING] Expected {EXPECTED_COLUMNS} columns, "
                f"got {num_cols}."
            )

        # Validate row count (signal should be ~100k)
        if num_rows < 1000:
            print(
                f"[ERROR] Too few events ({num_rows:,}). "
                "File may be corrupted."
            )
            return False

        if abs(num_rows - EXPECTED_SIGNAL_EVENTS) > 10_000:
            print(
                f"[WARNING] Expected ~{EXPECTED_SIGNAL_EVENTS:,} signal "
                f"events, got {num_rows:,}."
            )

        # Spot-check: sample a few events and verify pT > 0
        sample = node[0:5]
        sample_pt = sample[:, 0]  # First particle pT

        if np.all(sample_pt > 0):
            print("  Sample pT values valid (> 0).")
        else:
            print("[WARNING] Some sample pT values are <= 0.")

        # Check truth labels are all 1 (signal)
        truth_labels = node[:, -1]
        unique_labels = np.unique(truth_labels)
        print(f"  Unique truth labels: {unique_labels}")

        if len(unique_labels) == 1 and unique_labels[0] == 1.0:
            print("  ✓ All events are signal (truth == 1).")
        else:
            print("[WARNING] File contains non-signal events.")

    finally:
        h5file.close()

    print("[INFO] Verification passed. ✓")
    return True


########################################
# MAIN ENTRY POINT
########################################

if __name__ == "__main__":

    print("=" * 50)
    print("  LHC Olympics 2020 — W' Signal Downloader")
    print("=" * 50)
    print()
    print(f"  Zenodo Record: https://zenodo.org/records/{ZENODO_RECORD_ID}")
    print(f"  Output Path:   {SIGNAL_OUTPUT_PATH}")
    print()

    # Step 1: Download the full R&D dataset from Zenodo
    download_from_zenodo()

    # Step 2: Extract signal events (truth == 1) into separate file
    num_signal = extract_signal_events()

    if num_signal > 0:
        print(f"\n[INFO] Extracted {num_signal:,} signal events.")

    # Step 3: Verify the extracted signal file
    is_valid = verify_signal_file()

    if is_valid:
        print("\n[SUCCESS] W' signal dataset is ready for use.")
        print(f"  → {SIGNAL_OUTPUT_PATH}")
    else:
        print("\n[FAILURE] Signal file verification failed.")
        sys.exit(1)
