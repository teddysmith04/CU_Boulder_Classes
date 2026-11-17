#!/usr/bin/env python3
"""
Download COSMOS-Web SED plots (PDFs) for a list of IDs.

Usage:
    python download_sed_plots.py ids_above_color_cut.txt
    python download_sed_plots.py ids_above_color_cut.txt -o sed_plots --workers 8
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

URL_TEMPLATE = "https://cosmos2025.iap.fr/fitsmap/data/inspec_plots/cosmos_web_sed_{id}.pdf"


def read_ids(path):
    """Read one ID per line, ignoring blanks and duplicates (order preserved)."""
    seen, ids = set(), []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and s not in seen:
                seen.add(s)
                ids.append(s)
    return ids


def download_one(obj_id, out_dir, session, retries=3, timeout=30):
    """Download a single PDF. Returns (id, status, message)."""
    dest = out_dir / f"cosmos_web_sed_{obj_id}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return obj_id, "skipped", "already exists"

    url = URL_TEMPLATE.format(id=obj_id)
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 404:
                return obj_id, "missing", "404 not found"
            r.raise_for_status()

            # Write to a temp file first so a failed download never leaves a corrupt PDF
            tmp = dest.with_suffix(".pdf.part")
            tmp.write_bytes(r.content)
            tmp.rename(dest)
            return obj_id, "ok", f"{len(r.content) / 1024:.0f} KB"
        except requests.RequestException as e:
            if attempt == retries:
                return obj_id, "failed", str(e)
            time.sleep(2 * attempt)  # simple backoff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id_file", help="Text file with one ID per line")
    ap.add_argument("-o", "--out-dir", default="sed_plots", help="Output directory (default: sed_plots)")
    ap.add_argument("-w", "--workers", type=int, default=4, help="Parallel downloads (default: 4)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = read_ids(args.id_file)
    print(f"Found {len(ids)} unique IDs. Saving to {out_dir.resolve()}")

    counts = {"ok": 0, "skipped": 0, "missing": 0, "failed": 0}
    problems = []

    with requests.Session() as session, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, i, out_dir, session) for i in ids]
        for n, fut in enumerate(as_completed(futures), 1):
            obj_id, status, msg = fut.result()
            counts[status] += 1
            if status in ("missing", "failed"):
                problems.append((obj_id, status, msg))
            print(f"[{n}/{len(ids)}] {obj_id}: {status} ({msg})")

    print("\nSummary:", ", ".join(f"{k}={v}" for k, v in counts.items()))

    if problems:
        log = out_dir / "failed_ids.txt"
        log.write_text("\n".join(p[0] for p in problems) + "\n")
        print(f"{len(problems)} IDs had problems; list saved to {log}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())