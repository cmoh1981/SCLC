#!/usr/bin/env python
"""Download PRIDE proteomics data PXD052033."""

import os
import requests
import json
from pathlib import Path

DATA_DIR = Path("data/pride/PXD052033")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# PRIDE Archive REST API
PRIDE_API = "https://www.ebi.ac.uk/pride/ws/archive/v2"

def download_pride_metadata():
    """Download PXD052033 project metadata from PRIDE."""
    print("Downloading PRIDE PXD052033 metadata...")
    print("="*60)

    project_id = "PXD052033"

    # Get project metadata
    print("\nFetching project details...")
    project_url = f"{PRIDE_API}/projects/{project_id}"
    try:
        resp = requests.get(project_url)
        if resp.status_code == 200:
            data = resp.json()
            with open(DATA_DIR / "project_metadata.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"[OK] Project: {data.get('title', 'N/A')}")
            print(f"  Description: {data.get('projectDescription', 'N/A')[:100]}...")
        else:
            print(f"[ERROR] HTTP {resp.status_code}")
    except Exception as e:
        print(f"[ERROR] Error fetching project: {e}")

    # Get file list
    print("\nFetching file list...")
    files_url = f"{PRIDE_API}/projects/{project_id}/files"
    try:
        resp = requests.get(files_url)
        if resp.status_code == 200:
            data = resp.json()
            with open(DATA_DIR / "files_list.json", "w") as f:
                json.dump(data, f, indent=2)

            # Create download links file
            with open(DATA_DIR / "download_links.txt", "w") as f:
                f.write(f"# PRIDE Project: {project_id}\n")
                f.write("# Use aspera or FTP to download raw files\n\n")
                for file_info in data:
                    if 'publicFileLocations' in file_info:
                        for loc in file_info['publicFileLocations']:
                            f.write(f"{loc.get('value', 'N/A')}\n")
            print(f"[OK] Found {len(data)} files")
            print(f"  Download links saved to download_links.txt")
        else:
            print(f"[ERROR] HTTP {resp.status_code}")
    except Exception as e:
        print(f"[ERROR] Error fetching files: {e}")

    print("\n" + "="*60)
    print(f"[OK] PXD052033 metadata downloaded to {DATA_DIR}")
    print("\nNote: Raw proteomics files are large. Use Aspera or FTP client")
    print("to download specific files from the links in download_links.txt")

if __name__ == "__main__":
    download_pride_metadata()
