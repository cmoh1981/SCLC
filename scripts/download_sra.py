#!/usr/bin/env python
"""Download SRA metadata for SCLC projects."""

import os
import json
from pathlib import Path

try:
    from pysradb.sraweb import SRAweb
    HAS_PYSRADB = True
except ImportError:
    HAS_PYSRADB = False
    print("[WARN] pysradb not installed. Using Entrez fallback.")

from Bio import Entrez

# Set email for NCBI (required)
Entrez.email = "sclc-research@example.com"

DATA_DIR = Path("data/sra")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# SRA projects to fetch metadata for
SRA_PROJECTS = [
    {
        "accession": "PRJNA1014231",
        "description": "Zhang et al. Adv Sci 2024 - Spatial Transcriptomics of 25 SCLC tumors"
    },
    {
        "accession": "PRJNA575243",
        "description": "Stewart et al. Nature Cancer 2020 - scRNA-seq CTC-derived xenografts"
    }
]

def download_sra_metadata_pysradb(accession, description):
    """Download SRA project metadata using pysradb."""
    print(f"\n{'='*60}")
    print(f"Downloading {accession}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    dest_dir = DATA_DIR / accession
    dest_dir.mkdir(exist_ok=True)

    try:
        db = SRAweb()

        # Get project metadata
        df = db.sra_metadata(accession)

        if df is not None and len(df) > 0:
            # Save metadata as CSV
            csv_path = dest_dir / f"{accession}_metadata.csv"
            df.to_csv(csv_path, index=False)
            print(f"[OK] Saved {len(df)} run records to {csv_path}")

            # Save sample summary
            summary = {
                "accession": accession,
                "description": description,
                "total_runs": len(df),
                "columns": list(df.columns)
            }
            with open(dest_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            return True
        else:
            print(f"[WARN] No metadata found for {accession}")
            return False

    except Exception as e:
        print(f"[ERROR] {accession}: {e}")
        return False

def download_sra_metadata_entrez(accession, description):
    """Fallback: Get SRA project info via Entrez."""
    print(f"\n{'='*60}")
    print(f"Downloading {accession} (Entrez)")
    print(f"Description: {description}")
    print(f"{'='*60}")

    dest_dir = DATA_DIR / accession
    dest_dir.mkdir(exist_ok=True)

    try:
        # Search for the project
        handle = Entrez.esearch(db="sra", term=accession, retmax=1000)
        record = Entrez.read(handle)
        handle.close()

        ids = record.get("IdList", [])
        print(f"[OK] Found {len(ids)} SRA records")

        if ids:
            # Save run IDs
            with open(dest_dir / "sra_ids.txt", "w") as f:
                f.write("\n".join(ids))

            # Get summary for first few records
            handle = Entrez.esummary(db="sra", id=",".join(ids[:10]))
            summaries = Entrez.read(handle)
            handle.close()

            with open(dest_dir / "sample_summaries.json", "w") as f:
                json.dump(summaries, f, indent=2, default=str)

            summary = {
                "accession": accession,
                "description": description,
                "total_runs": len(ids),
                "method": "entrez"
            }
            with open(dest_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(f"[OK] Metadata saved to {dest_dir}")
            return True

        return False

    except Exception as e:
        print(f"[ERROR] {accession}: {e}")
        return False

if __name__ == "__main__":
    print("SCLC SRA Metadata Downloader")
    print("="*60)

    for project in SRA_PROJECTS:
        if HAS_PYSRADB:
            download_sra_metadata_pysradb(project["accession"], project["description"])
        else:
            download_sra_metadata_entrez(project["accession"], project["description"])

    print("\n" + "="*60)
    print("[INFO] SRA metadata downloaded.")
    print("[INFO] To download raw FASTQ files, use:")
    print("       prefetch <SRR_ACCESSION>")
    print("       fasterq-dump <SRR_ACCESSION>")
