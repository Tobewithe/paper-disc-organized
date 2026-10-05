"""Focused arXiv search for directly overlapping instance-mask calibration work."""

from __future__ import annotations

import json

import arxiv


QUERIES = [
    'all:"instance segmentation" AND all:"adaptive threshold"',
    'all:"instance segmentation" AND all:"mask calibration"',
    'all:"instance segmentation" AND all:"dynamic threshold"',
    'all:"instance segmentation" AND all:"mask quality" AND all:"scale"',
    'all:"prototype" AND all:"mask coefficient" AND all:"instance segmentation"',
]


def main():
    client = arxiv.Client(page_size=25, delay_seconds=3, num_retries=3)
    seen = {}
    for query in QUERIES:
        search = arxiv.Search(query=query, max_results=25, sort_by=arxiv.SortCriterion.Relevance)
        for paper in client.results(search):
            paper_id = paper.entry_id.rsplit("/", 1)[-1]
            if paper_id in seen:
                seen[paper_id]["queries"].append(query)
                continue
            seen[paper_id] = {
                "title": paper.title,
                "authors": [str(a) for a in paper.authors[:5]],
                "year": paper.published.year,
                "arxiv_id": paper_id,
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "categories": paper.categories,
                "summary": " ".join(paper.summary.split()),
                "queries": [query],
            }
    print(json.dumps(list(seen.values()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
