#!/usr/bin/env python3
"""Run local fallback searches with retries and source-level recovery.

This script is the no-MCP rescue path. It tries each local source, retries transient
failures, records source status, and keeps partial results instead of aborting the
whole search.

    python3 resilient_search.py "graph neural networks" -n 10 --status status.json -o raw.json
"""
import argparse
import json
import sys
import time

from search_local import DISPATCH


DEFAULT_SOURCES = ["arxiv", "semanticscholar", "ddg"]


def run_source(source, query, limit, retries, backoff):
    attempts = []
    for attempt in range(1, retries + 2):
        started = time.time()
        try:
            records = DISPATCH[source](query, limit)
            return records, {
                "source": source,
                "status": "ok" if records else "empty",
                "attempts": attempts + [{"attempt": attempt, "seconds": round(time.time() - started, 3)}],
                "records": len(records),
            }
        except Exception as e:  # noqa: BLE001 - this is a recovery boundary
            attempts.append({
                "attempt": attempt,
                "seconds": round(time.time() - started, 3),
                "error": str(e),
            })
            if attempt <= retries:
                time.sleep(backoff * attempt)
    return [], {"source": source, "status": "failed", "attempts": attempts, "records": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("-n", "--limit", type=int, default=20)
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help="comma-separated local sources: arxiv,semanticscholar,ddg")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--min-results", type=int, default=1,
                    help="warn in status if fewer records are recovered")
    ap.add_argument("-o", "--out")
    ap.add_argument("--status")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    bad = [s for s in sources if s not in DISPATCH]
    if bad:
        sys.exit(f"unknown source(s): {', '.join(bad)}")

    all_records = []
    statuses = []
    for source in sources:
        records, status = run_source(source, args.query, args.limit, args.retries, args.backoff)
        all_records.extend(records)
        statuses.append(status)
        print(f"{source}: {status['status']} ({status['records']} records)", file=sys.stderr)

    manifest = {
        "query": args.query,
        "sources": statuses,
        "records": len(all_records),
        "status": "ok" if len(all_records) >= args.min_results else "underfilled",
    }

    payload = json.dumps(all_records, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"wrote {len(all_records)} records -> {args.out}", file=sys.stderr)
    else:
        print(payload)

    if args.status:
        with open(args.status, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"wrote status -> {args.status}", file=sys.stderr)

    if manifest["status"] != "ok":
        sys.exit(3)


if __name__ == "__main__":
    main()
