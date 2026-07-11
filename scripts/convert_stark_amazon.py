# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the STaRK-Amazon SKB into relational CSVs for Postgres seeding.

Loads the processed Amazon knowledge base via ``stark_qa`` (requires Python
<3.12) and writes ``datasets/amazon/data/amazon.<table>.csv`` files compatible
with :mod:`scripts.seed_postgres`.

Usage (isolated env — stark-qa does not support Python 3.12+)::

    uv run --python 3.11 --with stark-qa python scripts/convert_stark_amazon.py

Optional flags::

    --output-dir PATH          Override output directory (default: datasets/amazon/data)
    --cache-dir PATH           Where stark_qa stores/downloads SKB data
    --max-reviews-per-product  Cap review rows per product (default: unlimited)
    --max-qa-per-product       Cap Q&A rows per product (default: unlimited)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _REPO_ROOT / "datasets" / "amazon" / "data"


def _null(value: Any) -> str:
    """Format a value for Postgres COPY (NULL -> ``\\N``)."""
    if value is None:
        return r"\N"
    if isinstance(value, bool):
        return "t" if value else "f"
    if isinstance(value, (list, dict)):
        if not value:
            return r"\N"
        return json.dumps(value, ensure_ascii=False)
    text = str(value)
    if text == "":
        return r"\N"
    return text


def _join_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if v]
        return " ".join(parts) if parts else None
    return str(value).strip() or None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _null(row.get(k)) for k in fieldnames})
    logger.info("Wrote %s (%d rows)", path.name, len(rows))


def convert(
    output_dir: Path,
    *,
    cache_dir: Path | None = None,
    max_reviews_per_product: int | None = None,
    max_qa_per_product: int | None = None,
    skip_reviews: bool = False,
    skip_qa: bool = False,
) -> None:
    from stark_qa import load_skb

    logger.info("Loading STaRK-Amazon SKB (this may download processed data)...")
    root = str(cache_dir) if cache_dir else None
    skb = load_skb("amazon", root=root, download_processed=True)

    num_nodes = skb.num_nodes()
    num_edges = skb.num_edges()
    logger.info(
        "SKB loaded: %d nodes, %d edges, node types=%s, edge types=%s",
        num_nodes,
        num_edges,
        skb.node_type_lst(),
        skb.rel_type_lst(),
    )

    products: list[dict[str, Any]] = []
    brands: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    colors: list[dict[str, Any]] = []
    product_categories: list[dict[str, Any]] = []
    product_colors: list[dict[str, Any]] = []
    also_buy: list[dict[str, Any]] = []
    also_view: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []

    product_brand: dict[int, int] = {}
    seen_product_categories: set[tuple[int, int]] = set()
    seen_product_colors: set[tuple[int, int]] = set()
    seen_also_buy: set[tuple[int, int]] = set()
    seen_also_view: set[tuple[int, int]] = set()

    edge_index = skb.edge_index
    for edge_idx in range(num_edges):
        src = int(edge_index[0, edge_idx].item())
        dst = int(edge_index[1, edge_idx].item())
        edge_type = skb.get_edge_type_by_id(edge_idx)
        src_type = skb.get_node_type_by_id(src)
        dst_type = skb.get_node_type_by_id(dst)

        if edge_type == "has_brand":
            if src_type == "product" and dst_type == "brand":
                product_brand[src] = dst
            elif src_type == "brand" and dst_type == "product":
                product_brand[dst] = src
        elif edge_type == "has_category":
            if src_type == "product" and dst_type == "category":
                pair = (src, dst)
            elif src_type == "category" and dst_type == "product":
                pair = (dst, src)
            else:
                continue
            if pair not in seen_product_categories:
                seen_product_categories.add(pair)
                product_categories.append(
                    {"product_id": pair[0], "category_id": pair[1]}
                )
        elif edge_type == "has_color":
            if src_type == "product" and dst_type == "color":
                pair = (src, dst)
            elif src_type == "color" and dst_type == "product":
                pair = (dst, src)
            else:
                continue
            if pair not in seen_product_colors:
                seen_product_colors.add(pair)
                product_colors.append({"product_id": pair[0], "color_id": pair[1]})
        elif edge_type == "also_buy":
            if src_type != "product" or dst_type != "product":
                continue
            if src != dst and (src, dst) not in seen_also_buy:
                seen_also_buy.add((src, dst))
                also_buy.append(
                    {"product_id": src, "related_product_id": dst}
                )
        elif edge_type == "also_view":
            if src_type != "product" or dst_type != "product":
                continue
            if src != dst and (src, dst) not in seen_also_view:
                seen_also_view.add((src, dst))
                also_view.append(
                    {"product_id": src, "related_product_id": dst}
                )

    review_id = 0
    qa_id = 0

    for node_id in range(num_nodes):
        node_type = skb.get_node_type_by_id(node_id)
        info = skb.node_info[node_id]

        if node_type == "brand":
            brands.append(
                {
                    "brand_id": node_id,
                    "brand_name": info.get("brand_name"),
                }
            )
        elif node_type == "category":
            categories.append(
                {
                    "category_id": node_id,
                    "category_name": info.get("category_name"),
                }
            )
        elif node_type == "color":
            colors.append(
                {
                    "color_id": node_id,
                    "color_name": info.get("color_name"),
                }
            )
        elif node_type == "product":
            details = info.get("details")
            if details is not None and not isinstance(details, str):
                details = json.dumps(details, ensure_ascii=False)

            products.append(
                {
                    "product_id": node_id,
                    "asin": info.get("asin"),
                    "title": info.get("title"),
                    "description": _join_text(info.get("description")),
                    "feature": _join_text(info.get("feature")),
                    "price": info.get("price"),
                    "rank": _join_text(info.get("rank")),
                    "global_category": info.get("global_category"),
                    "details": details,
                    "brand_id": product_brand.get(node_id),
                }
            )

            if not skip_reviews:
                product_reviews = info.get("review") or []
                if max_reviews_per_product is not None:
                    product_reviews = product_reviews[:max_reviews_per_product]
                for review in product_reviews:
                    review_id += 1
                    style = review.get("style")
                    if style is not None and not isinstance(style, str):
                        style = json.dumps(style, ensure_ascii=False)
                    overall = review.get("overall")
                    if overall is not None:
                        try:
                            overall = float(overall)
                        except (TypeError, ValueError):
                            overall = None
                    reviews.append(
                        {
                            "review_id": review_id,
                            "product_id": node_id,
                            "reviewer_id": review.get("reviewerID"),
                            "summary": review.get("summary"),
                            "review_text": review.get("reviewText"),
                            "vote": review.get("vote"),
                            "overall": overall,
                            "verified": review.get("verified"),
                            "review_time": review.get("reviewTime"),
                            "style": style,
                        }
                    )

            if not skip_qa:
                product_qa = info.get("qa") or []
                if max_qa_per_product is not None:
                    product_qa = product_qa[:max_qa_per_product]
                for qa_item in product_qa:
                    qa_id += 1
                    qa_rows.append(
                        {
                            "qa_id": qa_id,
                            "product_id": node_id,
                            "question_type": qa_item.get("questionType"),
                            "answer_type": qa_item.get("answerType"),
                            "question": qa_item.get("question"),
                            "answer": qa_item.get("answer"),
                            "answer_time": qa_item.get("answerTime"),
                        }
                    )

    tables: list[tuple[str, list[str], list[dict[str, Any]]]] = [
        (
            "products",
            [
                "product_id",
                "asin",
                "title",
                "description",
                "feature",
                "price",
                "rank",
                "global_category",
                "details",
                "brand_id",
            ],
            products,
        ),
        ("brands", ["brand_id", "brand_name"], brands),
        ("categories", ["category_id", "category_name"], categories),
        ("colors", ["color_id", "color_name"], colors),
        (
            "product_categories",
            ["product_id", "category_id"],
            product_categories,
        ),
        ("product_colors", ["product_id", "color_id"], product_colors),
        (
            "also_buy",
            ["product_id", "related_product_id"],
            also_buy,
        ),
        (
            "also_view",
            ["product_id", "related_product_id"],
            also_view,
        ),
    ]
    if not skip_reviews:
        tables.append(
            (
                "reviews",
                [
                    "review_id",
                    "product_id",
                    "reviewer_id",
                    "summary",
                    "review_text",
                    "vote",
                    "overall",
                    "verified",
                    "review_time",
                    "style",
                ],
                reviews,
            )
        )
    if not skip_qa:
        tables.append(
            (
                "qa",
                [
                    "qa_id",
                    "product_id",
                    "question_type",
                    "answer_type",
                    "question",
                    "answer",
                    "answer_time",
                ],
                qa_rows,
            )
        )

    for table_name, fieldnames, rows in tables:
        _write_csv(output_dir / f"amazon.{table_name}.csv", fieldnames, rows)

    logger.info(
        "Conversion complete: %d products, %d brands, %d categories, %d colors, "
        "%d also_buy, %d also_view, %d reviews, %d qa",
        len(products),
        len(brands),
        len(categories),
        len(colors),
        len(also_buy),
        len(also_view),
        len(reviews),
        len(qa_rows),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert STaRK-Amazon SKB to relational CSVs for Postgres seeding."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output directory for CSV files (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for stark_qa to download/cache SKB data",
    )
    parser.add_argument(
        "--max-reviews-per-product",
        type=int,
        default=None,
        help="Cap review rows exported per product (default: all)",
    )
    parser.add_argument(
        "--max-qa-per-product",
        type=int,
        default=None,
        help="Cap Q&A rows exported per product (default: all)",
    )
    parser.add_argument(
        "--skip-reviews",
        action="store_true",
        help="Skip exporting reviews (reuse existing amazon.reviews.csv)",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip exporting Q&A (reuse existing amazon.qa.csv)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    convert(
        args.output_dir,
        cache_dir=args.cache_dir,
        max_reviews_per_product=args.max_reviews_per_product,
        max_qa_per_product=args.max_qa_per_product,
        skip_reviews=args.skip_reviews,
        skip_qa=args.skip_qa,
    )


if __name__ == "__main__":
    main()
