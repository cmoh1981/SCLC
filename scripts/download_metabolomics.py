#!/usr/bin/env python
"""Download Metabolomics Workbench data ST000220."""

import os
import requests
from pathlib import Path

DATA_DIR = Path("data/metabolomics/ST000220")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Metabolomics Workbench REST API
BASE_URL = "https://www.metabolomicsworkbench.org/rest/study"

def download_metabolomics_data():
    """Download ST000220 study data from Metabolomics Workbench."""
    print("Downloading Metabolomics Workbench ST000220...")
    print("="*60)

    # Study summary
    print("\nFetching study summary...")
    summary_url = f"{BASE_URL}/study_id/ST000220/summary"
    try:
        resp = requests.get(summary_url)
        if resp.status_code == 200:
            with open(DATA_DIR / "study_summary.json", "w") as f:
                f.write(resp.text)
            print("[OK] Study summary saved")
    except Exception as e:
        print(f"[ERROR] Error fetching summary: {e}")

    # Analysis data
    print("\nFetching analysis data...")
    analysis_url = f"{BASE_URL}/study_id/ST000220/analysis"
    try:
        resp = requests.get(analysis_url)
        if resp.status_code == 200:
            with open(DATA_DIR / "analysis.json", "w") as f:
                f.write(resp.text)
            print("[OK] Analysis data saved")
    except Exception as e:
        print(f"[ERROR] Error fetching analysis: {e}")

    # Metabolites
    print("\nFetching metabolite data...")
    metabolites_url = f"{BASE_URL}/study_id/ST000220/metabolites"
    try:
        resp = requests.get(metabolites_url)
        if resp.status_code == 200:
            with open(DATA_DIR / "metabolites.json", "w") as f:
                f.write(resp.text)
            print("[OK] Metabolite data saved")
    except Exception as e:
        print(f"[ERROR] Error fetching metabolites: {e}")

    # Data matrix (mwtab format)
    print("\nFetching data matrix...")
    mwtab_url = f"{BASE_URL}/study_id/ST000220/mwtab"
    try:
        resp = requests.get(mwtab_url)
        if resp.status_code == 200:
            with open(DATA_DIR / "ST000220.mwtab", "w") as f:
                f.write(resp.text)
            print("[OK] mwTab data saved")
    except Exception as e:
        print(f"[ERROR] Error fetching mwtab: {e}")

    # Factors (sample metadata)
    print("\nFetching sample factors...")
    factors_url = f"{BASE_URL}/study_id/ST000220/factors"
    try:
        resp = requests.get(factors_url)
        if resp.status_code == 200:
            with open(DATA_DIR / "factors.json", "w") as f:
                f.write(resp.text)
            print("[OK] Sample factors saved")
    except Exception as e:
        print(f"[ERROR] Error fetching factors: {e}")

    print("\n" + "="*60)
    print(f"[OK] ST000220 data downloaded to {DATA_DIR}")

if __name__ == "__main__":
    download_metabolomics_data()
