#!/usr/bin/env python3
"""
07b_disgenet_evidence.py
- Hub genes -> DisGeNET Gene-Disease Association (GDA) evidence retrieval
- Filters to SCLC / lung cancer / immune resistance related disease terms
- Caches responses, handles rate limits, and writes paper-ready TSV + a network figure

Notes:
- DisGeNET API documentation is JS-heavy; endpoint paths/auth scheme can vary by plan/version.
- This script implements a robust "auto-probe" strategy:
  - tries multiple base URLs (env override) and multiple auth schemes (Bearer, X-Api-Key, apiKey param)
  - tries multiple common GDA endpoint path patterns
- If all probes fail, you only need to update DISGENET_BASE_URL and/or ENDPOINT_PATTERNS.

Reference: DisGeNET provides a REST API (base URL historically at http(s)://www.disgenet.org/api/). 
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from dotenv import load_dotenv

import networkx as nx
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------

DEFAULT_BASE_URL = os.environ.get("DISGENET_BASE_URL", "https://www.disgenet.org/api").rstrip("/")
API_KEY_ENV = "DISGENET_API_KEY"  # Environment variable name (set in .env file)

# Common disease term filters for your SCLC immunotherapy-resistance focus.
# You can edit this list without changing code.
DEFAULT_DISEASE_KEYWORDS = [
    "small cell lung", "sclc", "lung cancer", "pulmonary carcinoma", "neuroendocrine carcinoma",
    "immune", "immunotherapy", "checkpoint", "pd-1", "pd-l1", "ctla-4", "resistance"
]

# Common endpoint patterns observed across DisGeNET-era APIs and wrappers.
# We will probe these in order for each gene.
ENDPOINT_PATTERNS = [
    # Gene -> diseases (GDA)
    "/gda/gene/{gene}",                 # e.g., /gda/gene/TP53
    "/gda/gene/{gene}/disease",         # e.g., /gda/gene/TP53/disease
    "/gda/gene/{gene}?format=json",     # e.g., /gda/gene/TP53?format=json
    "/gda/gene/{gene}.json",            # e.g., /gda/gene/TP53.json
    # Some APIs distinguish by vocabulary/identifiers
    "/gda/gene/{gene}/ALL",             # e.g., /gda/gene/TP53/ALL
]

# Rate-limit / politeness
SLEEP_BETWEEN_CALLS_SEC = 0.15

# I/O
INPUT_HUB = "results/modules/hub_genes.tsv"
OUTDIR = "results/disgenet"
FIGDIR = "results/figures"

CACHE_DIR = os.path.join(OUTDIR, "cache")
MANIFEST_PATH = os.path.join(OUTDIR, "run_manifest.json")

OUT_ALL = os.path.join(OUTDIR, "hubgene_disease_evidence.tsv")
OUT_FILTERED = os.path.join(OUTDIR, "sclc_filtered_evidence.tsv")
FIG_PATH = os.path.join(FIGDIR, "Fig_disgenet_network.png")


# -----------------------------
# Utilities
# -----------------------------

def ensure_dirs() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_hub_genes(path: str) -> List[str]:
    df = pd.read_csv(path, sep="\t")
    for col in ["gene_symbol", "gene", "symbol"]:
        if col in df.columns:
            genes = df[col].astype(str).str.strip().tolist()
            genes = [g for g in genes if g and g != "nan"]
            return sorted(list(dict.fromkeys(genes)))
    raise ValueError(f"Input hub gene file must contain one of columns: gene_symbol/gene/symbol. Got: {list(df.columns)}")


def keyword_match(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


@dataclass
class DisGeNETClient:
    base_url: str
    api_key: str
    timeout: int = 30

    def _auth_variants(self) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
        """
        Returns list of (headers, params) auth variants to try.
        """
        k = self.api_key
        return [
            ({"Authorization": f"Bearer {k}"}, {}),      # Bearer token style
            ({"X-Api-Key": k}, {}),                      # API key header style
            ({}, {"apiKey": k}),                         # apiKey query param style
            ({}, {"apikey": k}),                         # alternate query param
        ]

    def _cache_path(self, key: str) -> str:
        return os.path.join(CACHE_DIR, f"{sha1(key)}.json")

    def _read_cache(self, key: str) -> Optional[Any]:
        p = self._cache_path(key)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _write_cache(self, key: str, payload: Any) -> None:
        p = self._cache_path(key)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=20),
        retry=retry_if_exception_type((requests.RequestException, RuntimeError)),
    )
    def _get_json(self, url: str, headers: Dict[str, str], params: Dict[str, str]) -> Any:
        r = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        if r.status_code in (429, 500, 502, 503, 504):
            raise RuntimeError(f"Transient HTTP {r.status_code} for {url}")
        if r.status_code == 401 or r.status_code == 403:
            # auth might be wrong; don't retry too much at this layer
            raise RuntimeError(f"Auth/Forbidden HTTP {r.status_code} for {url}")
        r.raise_for_status()
        try:
            return r.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON for {url}: {e}") from e

    def fetch_gene_gdas(self, gene: str) -> Tuple[str, Any]:
        """
        Try multiple endpoint patterns and auth variants. Returns (resolved_url, json_payload).
        Caches successful payload per (gene + resolved_url).
        """
        last_error = None
        gene_safe = gene.strip()

        for pat in ENDPOINT_PATTERNS:
            path = pat.format(gene=gene_safe)
            url = f"{self.base_url}{path}"

            # Cache per URL because different patterns return different schemas
            cache_key = f"{gene_safe}::{url}"
            cached = self._read_cache(cache_key)
            if cached is not None:
                return url, cached

            for headers, params in self._auth_variants():
                try:
                    payload = self._get_json(url, headers=headers, params=params)
                    self._write_cache(cache_key, payload)
                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                    return url, payload
                except Exception as e:
                    last_error = e
                    continue

        raise RuntimeError(f"All endpoint/auth probes failed for gene={gene_safe}. Last error: {last_error}")


# -----------------------------
# Parsing helpers (schema-robust)
# -----------------------------

def normalize_gda_rows(gene: str, source_url: str, payload: Any) -> pd.DataFrame:
    """
    Converts diverse DisGeNET payload schemas to a standard tabular format.
    We attempt multiple common field names. If schema mismatch, we store minimally.
    """
    rows: List[Dict[str, Any]] = []

    def push(row: Dict[str, Any]) -> None:
        row["query_gene"] = gene
        row["source_url"] = source_url
        rows.append(row)

    # Payload may be list[dict], dict with "results", dict with "data", etc.
    items = None
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for k in ["results", "data", "payload", "associations", "gda", "GDAs"]:
            if k in payload and isinstance(payload[k], list):
                items = payload[k]
                break
        if items is None:
            # maybe dict is a single association
            items = [payload]
    else:
        items = [{"raw": str(payload)}]

    for it in items:
        if not isinstance(it, dict):
            push({"raw": str(it)})
            continue

        # Try common keys for disease and scores
        disease_id = (
            it.get("diseaseid") or it.get("diseaseId") or it.get("disease_id") or
            it.get("disease_id_umls") or it.get("disease") or it.get("umls") or it.get("CUI")
        )
        disease_name = (
            it.get("diseasename") or it.get("diseaseName") or it.get("disease_name") or
            it.get("disease") if isinstance(it.get("disease"), str) else None
        )

        score = it.get("score") or it.get("disgenetScore") or it.get("gdaScore") or it.get("DS") or it.get("disgenet_score")
        n_pmids = it.get("n_pmids") or it.get("NPMIDS") or it.get("nPMIDs") or it.get("pmid_count") or it.get("NofPmids")
        source = it.get("source") or it.get("database") or it.get("sourceDatabase") or it.get("SOURCE")

        # Some schemas have nested disease objects
        if isinstance(it.get("disease"), dict):
            d = it["disease"]
            disease_id = disease_id or d.get("diseaseId") or d.get("id") or d.get("CUI")
            disease_name = disease_name or d.get("diseaseName") or d.get("name")

        push({
            "gene": gene,
            "disease_id": disease_id,
            "disease_name": disease_name,
            "score": score,
            "evidence_count": n_pmids,
            "source": source,
            "raw": it,  # keep raw for traceability (can be dropped in final)
        })

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    load_dotenv()

    ensure_dirs()

    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise SystemExit(f"Missing {API_KEY_ENV}. Put it in .env (NOT committed).")

    base_url = os.environ.get("DISGENET_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    client = DisGeNETClient(base_url=base_url, api_key=api_key)

    genes = load_hub_genes(INPUT_HUB)

    all_dfs = []
    failures = []

    for g in genes:
        try:
            resolved_url, payload = client.fetch_gene_gdas(g)
            df = normalize_gda_rows(g, resolved_url, payload)
            all_dfs.append(df)
        except Exception as e:
            failures.append({"gene": g, "error": str(e)})

    if not all_dfs:
        raise SystemExit(f"No successful DisGeNET queries. Failures: {failures[:3]} ...")

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Make raw JSON column serializable (optional). Keep raw as JSON string to avoid TSV issues.
    if "raw" in df_all.columns:
        df_all["raw"] = df_all["raw"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x))

    # Filter to SCLC/lung cancer/immune-resistance related disease names
    keywords = DEFAULT_DISEASE_KEYWORDS
    df_all["disease_name_str"] = df_all["disease_name"].fillna("").astype(str)
    df_filt = df_all[df_all["disease_name_str"].apply(lambda t: keyword_match(t, keywords))].copy()

    # Write outputs
    df_all.drop(columns=["disease_name_str"], errors="ignore").to_csv(OUT_ALL, sep="\t", index=False)
    df_filt.drop(columns=["disease_name_str"], errors="ignore").to_csv(OUT_FILTERED, sep="\t", index=False)

    # Build a simple network figure (gene -> disease)
    G = nx.Graph()
    for _, r in df_filt.iterrows():
        gene = str(r.get("gene") or r.get("query_gene"))
        disease = str(r.get("disease_name") or r.get("disease_id") or "NA")
        if disease == "NA":
            continue
        score = r.get("score")
        try:
            w = float(score) if score is not None and str(score) != "nan" else 0.0
        except Exception:
            w = 0.0
        G.add_node(gene, kind="gene")
        G.add_node(disease, kind="disease")
        G.add_edge(gene, disease, weight=w)

    if G.number_of_edges() > 0:
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=7, k=0.6)
        # Draw genes/diseases separately for legibility
        genes_n = [n for n, a in G.nodes(data=True) if a.get("kind") == "gene"]
        dis_n = [n for n, a in G.nodes(data=True) if a.get("kind") == "disease"]
        nx.draw_networkx_nodes(G, pos, nodelist=genes_n, node_size=450)
        nx.draw_networkx_nodes(G, pos, nodelist=dis_n, node_size=220)
        nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=7)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIG_PATH, dpi=300)
        plt.close()

    # Manifest for reproducibility
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "disgenet_base_url": base_url,
        "n_genes": len(genes),
        "n_rows_all": int(df_all.shape[0]),
        "n_rows_filtered": int(df_filt.shape[0]),
        "failures": failures,
        "keywords": keywords,
        "endpoint_patterns": ENDPOINT_PATTERNS,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {OUT_ALL}")
    print(f"[OK] Wrote: {OUT_FILTERED}")
    if os.path.exists(FIG_PATH):
        print(f"[OK] Wrote: {FIG_PATH}")
    print(f"[OK] Wrote: {MANIFEST_PATH}")
    if failures:
        print(f"[WARN] Failures for {len(failures)} genes (see manifest).")


if __name__ == "__main__":
    main()
