"""
download_datasets.py — Automated Dataset Acquisition & Symlinking
=================================================================
Safely discovers or downloads the LHCO2020 scientific datasets:
1. events_LHCO2020_backgroundMC_Pythia.h5 (~2.5 GB)
2. W_prime_signal.h5 (events_anomalydetection.h5, ~2.6 GB)

Usage:
  python scripts/download_datasets.py
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BG_TARGET = DATA_DIR / "events_LHCO2020_backgroundMC_Pythia.h5"
SIG_TARGET = DATA_DIR / "W_prime_signal.h5"

MIN_SIZE_BYTES = 100_000_000  # 100 MB minimum threshold

BG_URL = "https://zenodo.org/api/records/4536624/files/events_LHCO2020_backgroundMC_Pythia.h5/content"
SIG_URL = "https://zenodo.org/api/records/4536377/files/events_anomalydetection.h5/content"


def is_valid(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > MIN_SIZE_BYTES
    except Exception:
        return False


def find_in_candidates(patterns, search_roots):
    for root in search_roots:
        r_path = Path(root)
        if not r_path.exists():
            continue
        for pat in patterns:
            for match in r_path.glob(pat):
                if is_valid(match):
                    return match
    return None


def download_file(url: str, dest_path: Path, name: str) -> bool:
    print(f"\n[DOWNLOAD] Downloading {name} to {dest_path}...")
    print(f"URL: {url}")
    # Try wget first for fast resumable datacenter download with progress
    cmd = ["wget", "-c", "--show-progress", "-O", str(dest_path), url]
    try:
        ret = subprocess.run(cmd, check=True)
        if is_valid(dest_path):
            print(f"[SUCCESS] {name} downloaded ({dest_path.stat().st_size / (1024**3):.2f} GB)")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[INFO] wget failed or not installed. Falling back to curl...")
        curl_cmd = ["curl", "-L", "-C", "-", "-o", str(dest_path), url]
        try:
            subprocess.run(curl_cmd, check=True)
            if is_valid(dest_path):
                print(f"[SUCCESS] {name} downloaded via curl ({dest_path.stat().st_size / (1024**3):.2f} GB)")
                return True
        except Exception as e:
            print(f"[ERROR] curl failed: {e}")

    # Fallback to python urllib if needed
    if not is_valid(dest_path):
        print("[INFO] Falling back to urllib streaming...")
        import urllib.request
        try:
            with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
                while True:
                    chunk = response.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
            if is_valid(dest_path):
                print(f"[SUCCESS] {name} downloaded via urllib ({dest_path.stat().st_size / (1024**3):.2f} GB)")
                return True
        except Exception as e:
            print(f"[ERROR] urllib download failed: {e}")

    return is_valid(dest_path)


def main():
    print("=" * 80)
    print("  LHCO 2020 DATASET ACQUISITION & VERIFICATION")
    print(f"  Target directory: {DATA_DIR}")
    print("=" * 80)

    bg_target = BG_TARGET
    sig_target = SIG_TARGET

    search_roots = [
        "/kaggle/input",
        str(REPO_ROOT.parent),
        str(Path.home() / "data"),
    ]

    # 1. Background Dataset
    if is_valid(bg_target):
        print(f"[READY] Background dataset already verified: {bg_target} ({bg_target.stat().st_size / (1024**3):.2f} GB)")
    else:
        found_bg = find_in_candidates(
            ["**/*background*Pythia*.h5", "**/*background*.h5", "**/events_LHCO2020_backgroundMC_Pythia.h5"],
            search_roots,
        )
        if found_bg:
            print(f"[DISCOVERED] Found Background at {found_bg}")
            if bg_target.is_symlink() or bg_target.exists():
                bg_target.unlink()
            try:
                bg_target.symlink_to(found_bg)
                print(f"[LINKED] Symlinked {found_bg} -> {bg_target}")
            except Exception:
                bg_target = found_bg
        else:
            download_file(BG_URL, bg_target, "Background MC")

    # 2. Signal Dataset
    if is_valid(sig_target):
        print(f"[READY] Signal dataset already verified: {sig_target} ({sig_target.stat().st_size / (1024**3):.2f} GB)")
    else:
        found_sig = find_in_candidates(
            ["**/*W_prime*.h5", "**/*anomalydetection*.h5", "**/events_anomalydetection.h5"],
            search_roots,
        )
        if found_sig:
            print(f"[DISCOVERED] Found Signal at {found_sig}")
            if sig_target.is_symlink() or sig_target.exists():
                sig_target.unlink()
            try:
                sig_target.symlink_to(found_sig)
                print(f"[LINKED] Symlinked {found_sig} -> {sig_target}")
            except Exception:
                sig_target = found_sig
        else:
            download_file(SIG_URL, sig_target, "W' Signal / R&D")

    # Final summary
    bg_ok = is_valid(bg_target)
    sig_ok = is_valid(sig_target)

    print("\n" + "=" * 80)
    print("DATASET STATUS SUMMARY:")
    print(f"  Background (events_LHCO2020_backgroundMC_Pythia.h5): {'[OK]' if bg_ok else '[MISSING]'}")
    if bg_ok:
        print(f"    Path: {bg_target} ({bg_target.stat().st_size / (1024**3):.2f} GB)")
    print(f"  Signal (W_prime_signal.h5):                           {'[OK]' if sig_ok else '[MISSING]'}")
    if sig_ok:
        print(f"    Path: {sig_target} ({sig_target.stat().st_size / (1024**3):.2f} GB)")
    print("=" * 80)

    if not bg_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
