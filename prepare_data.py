#!/usr/bin/env python3
"""
SLURP Data Preparation Script

Downloads and extracts SLURP audio data from Zenodo.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

SLURP_REPO_URL = "https://github.com/pswietojanski/slurp"
ZENODO_BASE = "https://zenodo.org/record/4274930/files"
AUDIO_FILES = {
    "slurp_real": f"{ZENODO_BASE}/slurp_real.tar.gz",
    "slurp_synth": f"{ZENODO_BASE}/slurp_synth.tar.gz",
}


def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  {percent:3d}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()


def ensure_slurp_repo(slurp_dir: Path, skip_clone: bool = False) -> None:
    """Clone SLURP repo if not present."""
    dataset_dir = slurp_dir / "dataset" / "slurp"
    if dataset_dir.exists():
        print(f"✓ SLURP dataset already exists at {dataset_dir}")
        return
    if skip_clone:
        raise FileNotFoundError(f"SLURP dataset not found at {dataset_dir}")
    
    print(f"Cloning SLURP repository to {slurp_dir}...")
    if slurp_dir.exists():
        shutil.rmtree(slurp_dir)
    subprocess.run(["git", "clone", SLURP_REPO_URL, str(slurp_dir)], check=True)
    print("✓ SLURP repository cloned")


def download_audio(
    slurp_dir: Path,
    real_only: bool = False,
    skip_download: bool = False,
) -> None:
    """Download audio tar files from Zenodo."""
    audio_dir = slurp_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = ["slurp_real"] if real_only else list(AUDIO_FILES.keys())

    for name in files_to_download:
        tar_path = audio_dir / f"{name}.tar.gz"
        url = AUDIO_FILES[name]

        if tar_path.exists():
            print(f"✓ {tar_path.name} already exists ({tar_path.stat().st_size / 1e9:.2f} GB)")
            continue

        if skip_download:
            print(f"⚠ {tar_path.name} not found (skipping download)")
            continue

        print(f"Downloading {name}.tar.gz from Zenodo...")
        try:
            urlretrieve(url, tar_path, reporthook=progress_hook)
            print(f"\n✓ Downloaded {tar_path.name}")
        except Exception as e:
            print(f"\n✗ Failed to download {name}: {e}")
            if tar_path.exists():
                tar_path.unlink()
            raise


def extract_audio(
    slurp_dir: Path,
    real_only: bool = False,
    skip_extract: bool = False,
) -> None:
    """Extract audio tar files."""
    audio_dir = slurp_dir / "audio"

    files_to_extract = ["slurp_real"] if real_only else list(AUDIO_FILES.keys())

    for name in files_to_extract:
        tar_path = audio_dir / f"{name}.tar.gz"
        extract_dir = audio_dir / name

        if extract_dir.exists() and any(extract_dir.iterdir()):
            file_count = sum(1 for _ in extract_dir.rglob("*.flac"))
            print(f"✓ {name}/ already extracted ({file_count} files)")
            continue

        if skip_extract:
            print(f"⚠ Skipping extraction of {name}")
            continue

        if not tar_path.exists():
            print(f"⚠ {tar_path.name} not found, cannot extract")
            continue

        print(f"Extracting {name}.tar.gz...")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=audio_dir)
            file_count = sum(1 for _ in extract_dir.rglob("*.flac"))
            print(f"✓ Extracted {name}/ ({file_count} files)")
        except Exception as e:
            print(f"✗ Failed to extract {name}: {e}")
            raise


def validate_data(slurp_dir: Path, real_only: bool = False) -> bool:
    """Validate that audio files are properly extracted."""
    audio_dir = slurp_dir / "audio"
    datasets = ["slurp_real"] if real_only else ["slurp_real", "slurp_synth"]
    
    all_valid = True
    total_files = 0

    for name in datasets:
        extract_dir = audio_dir / name
        if not extract_dir.exists():
            print(f"✗ {name}/ directory not found")
            all_valid = False
            continue
        
        file_count = sum(1 for _ in extract_dir.rglob("*.flac"))
        total_files += file_count
        if file_count == 0:
            print(f"✗ {name}/ contains no .flac files")
            all_valid = False
        else:
            print(f"✓ {name}/: {file_count} audio files")

    if all_valid:
        print(f"\n✓ Data validation passed ({total_files} total audio files)")
    else:
        print("\n✗ Data validation failed")

    return all_valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SLURP data for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and extract all audio (real + synth, ~10GB)
  python prepare_data.py

  # Download only slurp_real (~4GB)
  python prepare_data.py --real-only

  # Only extract (if tar files already downloaded)
  python prepare_data.py --skip-download

  # Only download (skip extraction)
  python prepare_data.py --skip-extract
""",
    )
    parser.add_argument(
        "--slurp-dir",
        type=str,
        default="slurp",
        help="Path to SLURP directory (default: slurp)",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Only download/extract slurp_real (smaller, ~4GB)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading tar files",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extracting tar files",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning SLURP repo (assume it exists)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing data",
    )

    args = parser.parse_args()
    slurp_dir = Path(args.slurp_dir).resolve()

    print(f"SLURP directory: {slurp_dir}")
    print()

    if args.validate_only:
        validate_data(slurp_dir, args.real_only)
        return

    # Step 1: Ensure SLURP repo is cloned
    ensure_slurp_repo(slurp_dir, args.skip_clone)
    print()

    # Step 2: Download audio
    if not args.skip_download:
        download_audio(slurp_dir, args.real_only, skip_download=False)
    else:
        print("Skipping download step")
    print()

    # Step 3: Extract audio
    if not args.skip_extract:
        extract_audio(slurp_dir, args.real_only, skip_extract=False)
    else:
        print("Skipping extraction step")
    print()

    # Step 4: Validate
    validate_data(slurp_dir, args.real_only)


if __name__ == "__main__":
    main()
