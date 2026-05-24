"""
score_reviews.py
----------------
Scores restaurant reviews using the fine-tuned Eatsplorer ABSA model.

Reads reviews from:   eatsplorer.db  →  table: restaurant_reviews
                      columns: restaurant_name, rating, review_text

Writes scored rows to: eatsplorer.db  →  table: restaurant_reviews_scored
                      (created/replaced on each run)
  columns:
    restaurant_name, review_text, rating_overall,
    food_quality_score, food_quality_polarity,
    service_score,      service_polarity,
    ambiance_score,     ambiance_polarity,
    price_value_score,  price_value_polarity

Usage:
    python score_reviews.py
    python score_reviews.py --db path/to/eatsplorer.db
    python score_reviews.py --db eatsplorer.db --model ./eatsplorer_finetuned_model
"""

import argparse
import os
import sys
import sqlite3
import emoji
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -- config ------------------------------------------------------------------

DB_FILE   = "eatsplorer_populatedwithdates.db"
MODEL_DIR = "./eatsplorer_finetuned_model"

ASPECTS = ["food quality", "service", "ambiance", "price value", "overall"]

IDX_NA       = 0
IDX_NEGATIVE = 1
IDX_NEUTRAL  = 2
IDX_POSITIVE = 3

# -- helpers -----------------------------------------------------------------

def score_to_polarity(score):
    if score is None:
        return "N/A"
    if score >= 3.5:
        return "Positive"
    if score >= 2.6:
        return "Neutral"
    return "Negative"

def preprocess(text, rating):
    return f"[RATING={rating}] {emoji.demojize(str(text))}"

def compute_score(prob_array):
    prob_na  = float(prob_array[IDX_NA])
    prob_neg = float(prob_array[IDX_NEGATIVE])
    prob_neu = float(prob_array[IDX_NEUTRAL])
    prob_pos = float(prob_array[IDX_POSITIVE])

    if prob_na >= max(prob_neg, prob_neu, prob_pos):
        return None, "N/A"

    denom = prob_pos + prob_neu + prob_neg
    if denom == 0:
        return None, "N/A"

    score = (prob_pos * 5.0 + prob_neu * 3.0 + prob_neg * 1.0) / denom
    return round(score, 4), score_to_polarity(score)

# -- model loading -----------------------------------------------------------

def load_model(model_dir):
    if not os.path.exists(model_dir):
        print(f"\nERROR: model folder '{model_dir}' not found.")
        print("       Run train_masterv3.py first.")
        sys.exit(1)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()

    device_label = (
        f"GPU ({torch.cuda.get_device_name(0)})"
        if device.type == "cuda" else "CPU"
    )
    print(f"   model  : {model_dir}")
    print(f"   device : {device_label}\n")
    return model, tokenizer, device

# -- inference ---------------------------------------------------------------

def score_batch(model, tokenizer, device, aspects, texts, batch_size=32):
    results = []
    for start in range(0, len(texts), batch_size):
        batch_aspects = aspects[start:start + batch_size]
        batch_texts   = texts[start:start + batch_size]

        enc = tokenizer(
            text=batch_aspects,
            text_pair=batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

        for i in range(len(batch_aspects)):
            results.append(compute_score(probs[i]))

    return results

# -- main --------------------------------------------------------------------

def score_reviews(db_path, model_dir):
    if not os.path.exists(db_path):
        print(f"ERROR: database '{db_path}' not found.")
        sys.exit(1)

    # Read reviews from SQLite
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT restaurant_name, rating AS rating_overall, review_text, Year, Month FROM restaurant_reviews",
        conn
    )
    conn.close()

    print(f"reviews    : {len(df)}")
    print(f"restaurants: {df['restaurant_name'].nunique()}\n")

    model, tokenizer, device = load_model(model_dir)

    # Build flat lists — 4 aspect rows per review
    flat_aspects = []
    flat_texts   = []
    for _, row in df.iterrows():
        processed = preprocess(row["review_text"], row["rating_overall"])
        for aspect in ASPECTS:
            flat_aspects.append(aspect)
            flat_texts.append(processed)

    print(f"running inference on {len(flat_texts)} aspect-review pairs...")
    all_results = score_batch(model, tokenizer, device, flat_aspects, flat_texts)

    # Map results back to columns
    col_map = {
        "food quality": ("food_quality_score", "food_quality_polarity"),
        "service":      ("service_score",      "service_polarity"),
        "ambiance":     ("ambiance_score",      "ambiance_polarity"),
        "price value":  ("price_value_score",   "price_value_polarity"),
        "overall":      ("overall_score",       "overall_polarity"),
    }
    for score_col, polarity_col in col_map.values():
        df[score_col]    = None
        df[polarity_col] = "N/A"

    result_idx = 0
    for row_i in range(len(df)):
        for aspect in ASPECTS:
            score, polarity = all_results[result_idx]
            score_col, polarity_col = col_map[aspect]
            df.at[row_i, score_col]    = score
            df.at[row_i, polarity_col] = polarity
            result_idx += 1

    # Write scored rows back to SQLite (replace table each run)
    conn = sqlite3.connect(db_path)
    df.to_sql("restaurant_reviews_scored", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\ndone — wrote {len(df)} rows to table 'restaurant_reviews_scored' in {db_path}")

    for aspect in ASPECTS:
        pol_col = col_map[aspect][1]
        c = df[pol_col].value_counts()
        print(f"  {aspect:<14}: "
              f"Positive={c.get('Positive',0)}  "
              f"Neutral={c.get('Neutral',0)}  "
              f"Negative={c.get('Negative',0)}  "
              f"N/A={c.get('N/A',0)}")


def main():
    parser = argparse.ArgumentParser(
        description="Score restaurant reviews with the Eatsplorer ABSA model"
    )
    parser.add_argument("--db",    default=DB_FILE,   help="Path to eatsplorer.db")
    parser.add_argument("--model", default=MODEL_DIR, help="Path to fine-tuned model dir")
    args = parser.parse_args()

    print("\nEATSPLORER — score reviews\n")
    score_reviews(args.db, args.model)

if __name__ == "__main__":
    main()