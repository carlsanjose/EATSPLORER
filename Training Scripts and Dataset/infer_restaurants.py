"""
infer_restaurants.py
--------------------
Aggregates per-review ABSA scores up to the restaurant level.

Reads from:  eatsplorer.db  →  table: restaurant_reviews_scored
             (produced by score_reviews.py)

Writes to:   eatsplorer.db  →  table: restaurant_scores
             (created/replaced on each run)

Output columns in restaurant_scores:
    restaurant_name,
    food_quality_avg, food_quality_polarity, food_quality_review_count,
    service_avg,      service_polarity,      service_review_count,
    ambiance_avg,     ambiance_polarity,     ambiance_review_count,
    price_value_avg,  price_value_polarity,  price_value_review_count,
    overall_score,    overall_polarity,
    total_reviews,    aspects_scored

Usage:
    python infer_restaurants.py
    python infer_restaurants.py --db path/to/eatsplorer.db
"""

import argparse
import os
import sys
import sqlite3
import pandas as pd

# -- config ------------------------------------------------------------------

DB_FILE = "eatsplorer_populatedwithdates.db"
ASPECTS = ["food_quality", "service", "ambiance", "price_value", "overall"]

# -- helpers -----------------------------------------------------------------

def score_to_polarity(score):
    if score is None:
        return "N/A"
    if score >= 3.5:
        return "Positive"
    if score >= 2.6:
        return "Neutral"
    return "Negative"

# -- aggregation -------------------------------------------------------------

def aggregate(df, group_keys):
    """
    Aggregates aspect scores for any combination of group_keys.
    group_keys examples:
        ["restaurant_name"]                     -> overall aggregation
        ["restaurant_name", "Year", "Month"]    -> monthly aggregation
    """
    rows = []
    for group_vals, group in df.groupby(group_keys):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        entry = dict(zip(group_keys, group_vals))
        entry["total_reviews"] = len(group)

        valid_avgs = []
        for aspect in ASPECTS:
            score_col = f"{aspect}_score"
            valid = pd.to_numeric(group[score_col], errors="coerce").dropna()
            if len(valid) == 0:
                entry[f"{aspect}_avg"]          = None
                entry[f"{aspect}_polarity"]      = "N/A"
                entry[f"{aspect}_review_count"]  = 0
            else:
                avg = round(valid.mean(), 4)
                entry[f"{aspect}_avg"]          = avg
                entry[f"{aspect}_polarity"]      = score_to_polarity(avg)
                entry[f"{aspect}_review_count"]  = len(valid)
                valid_avgs.append(avg)

        n = len(valid_avgs)
        if n == 0:
            entry["composite_score"]    = None
            entry["composite_polarity"] = "N/A"
        else:
            composite = round(sum(valid_avgs) / n, 4)
            entry["composite_score"]    = composite
            entry["composite_polarity"] = score_to_polarity(composite)

        entry["aspects_scored"] = n
        rows.append(entry)
    return rows

def infer(df):
    return aggregate(df, ["restaurant_name"])

def infer_monthly(df):
    return aggregate(df, ["restaurant_name", "Year", "Month"])

# -- console summary ---------------------------------------------------------

def print_summary(rows):
    print("\n" + "=" * 88)
    print("  RESTAURANT SCORES")
    print("=" * 88)
    print(
        f"  {'Restaurant':<30} {'Food':>6} {'Svc':>6} "
        f"{'Amb':>6} {'Price':>6} {'Overall':>8} {'Composite':>10} {'n':>3} {'Reviews':>7}"
    )
    print("  " + "-" * 93)

    def fmt(v):
        return f"{v:.2f}" if v is not None else "  N/A"

    for r in sorted(rows, key=lambda x: x["composite_score"] or 0, reverse=True):
        print(
            f"  {r['restaurant_name']:<30} "
            f"{fmt(r['food_quality_avg']):>6} "
            f"{fmt(r['service_avg']):>6} "
            f"{fmt(r['ambiance_avg']):>6} "
            f"{fmt(r['price_value_avg']):>6} "
            f"{fmt(r['overall_avg']):>8} "
            f"{fmt(r['composite_score']):>10} "
            f"{r['aspects_scored']:>3} "
            f"{r['total_reviews']:>7}"
        )

    print("  " + "-" * 93)
    print("  Scores on a 1-5 scale  |  n = aspects with at least one mention  |  Composite = avg of all 5 aspects")
    print("=" * 88)

# -- main --------------------------------------------------------------------

def infer_restaurants(db_path):
    if not os.path.exists(db_path):
        print(f"ERROR: database '{db_path}' not found.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    # Check that restaurant_reviews_scored exists
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()

    if "restaurant_reviews_scored" not in tables:
        print("ERROR: table 'restaurant_reviews_scored' not found in database.")
        print("       Run score_reviews.py first.")
        conn.close()
        sys.exit(1)

    df = pd.read_sql_query("SELECT * FROM restaurant_reviews_scored", conn)

    required = {"restaurant_name"} | {f"{a}_score" for a in ASPECTS}
    missing  = required - set(df.columns)
    if missing:
        print(f"ERROR: scored table is missing columns: {missing}")
        conn.close()
        sys.exit(1)

    print(f"reviews    : {len(df)}")
    print(f"restaurants: {df['restaurant_name'].nunique()}\n")
    conn.close()

    rows         = infer(df)
    monthly_rows = infer_monthly(df)

    # -- overall table --------------------------------------------------------
    col_order = (
        ["restaurant_name"] +
        [f"{a}_{s}" for a in ASPECTS for s in ("avg", "polarity", "review_count")] +
        ["composite_score", "composite_polarity", "total_reviews", "aspects_scored"]
    )
    out = pd.DataFrame(rows)[col_order]
    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # -- monthly table --------------------------------------------------------
    monthly_col_order = (
        ["restaurant_name", "Year", "Month"] +
        [f"{a}_{s}" for a in ASPECTS for s in ("avg", "polarity", "review_count")] +
        ["composite_score", "composite_polarity", "total_reviews", "aspects_scored"]
    )
    monthly_out = pd.DataFrame(monthly_rows)[monthly_col_order]
    monthly_out = monthly_out.sort_values(
        ["Year", "Month", "composite_score"], ascending=[True, True, False]
    ).reset_index(drop=True)

    # -- write to DB ----------------------------------------------------------
    conn = sqlite3.connect(db_path)
    out.to_sql("restaurant_scores", conn, if_exists="replace", index=False)
    monthly_out.to_sql("restaurant_scores_monthly", conn, if_exists="replace", index=False)
    conn.close()

    print_summary(rows)
    print(f"\ndone — wrote {len(out)} rows to 'restaurant_scores'")
    print(f"       wrote {len(monthly_out)} rows to 'restaurant_scores_monthly' in {db_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate review scores to restaurant-level scores in SQLite"
    )
    parser.add_argument("--db", default=DB_FILE, help="Path to eatsplorer.db")
    args = parser.parse_args()

    print("\nEATSPLORER — restaurant inference\n")
    infer_restaurants(args.db)

if __name__ == "__main__":
    main()