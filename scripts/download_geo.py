#!/usr/bin/env python
"""Download GEO datasets for SCLC project."""

import os
import GEOparse
from pathlib import Path

# GEO datasets to download
GEO_DATASETS = [
    "GSE60052",   # Bulk RNA-seq 79 SCLC tumors
    "GSE138267",  # scRNA-seq resistance (subseries)
    "GSE138474",  # scRNA-seq parent series
    "GSE267310",  # Spatial + proteomics
]

DATA_DIR = Path("data/geo")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_geo_dataset(accession):
    """Download a GEO dataset."""
    print(f"\n{'='*60}")
    print(f"Downloading {accession}...")
    print(f"{'='*60}")

    dest_dir = DATA_DIR / accession
    dest_dir.mkdir(exist_ok=True)

    try:
        # Download the GEO series
        gse = GEOparse.get_GEO(geo=accession, destdir=str(dest_dir), silent=False)

        # Print metadata
        print(f"\nTitle: {gse.metadata.get('title', ['N/A'])[0]}")
        print(f"Type: {gse.metadata.get('type', ['N/A'])[0]}")
        print(f"Platform(s): {list(gse.gpls.keys())}")
        print(f"Samples: {len(gse.gsms)}")

        # Save metadata summary
        with open(dest_dir / "metadata_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Accession: {accession}\n")
            f.write(f"Title: {gse.metadata.get('title', ['N/A'])[0]}\n")
            f.write(f"Summary: {gse.metadata.get('summary', ['N/A'])[0]}\n")
            f.write(f"Type: {gse.metadata.get('type', ['N/A'])[0]}\n")
            f.write(f"Platform(s): {list(gse.gpls.keys())}\n")
            f.write(f"Number of samples: {len(gse.gsms)}\n")
            f.write(f"\nSupplementary files:\n")
            for supp in gse.metadata.get('supplementary_file', []):
                f.write(f"  - {supp}\n")

        print(f"[OK] {accession} metadata downloaded to {dest_dir}")
        return True

    except Exception as e:
        print(f"[ERROR] Error downloading {accession}: {e}")
        return False

if __name__ == "__main__":
    print("SCLC GEO Dataset Downloader")
    print("="*60)

    results = {}
    for acc in GEO_DATASETS:
        results[acc] = download_geo_dataset(acc)

    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for acc, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {acc}")
