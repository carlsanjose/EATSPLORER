"""
db_query_helpers.py
-------------------
Helper utilities for matching user-supplied cuisine / signature_dish queries
against database values that are stored with comma (,) and slash (/) separators.

Example DB row value:  "Bicol Express / Laing, Pinangat"
User query:            "bicol express"   →  should MATCH

Usage
-----
Import and call from any action that queries the DB:

    from db_query_helpers import normalize_token, tokenize_field, field_matches_query

Then in your SQL result loop:

    if field_matches_query(row["signature_dish"], slot_value):
        ...
"""

import re
import unicodedata


# ─────────────────────────────────────────────────────────────────────────────
# Core normalizer
# ─────────────────────────────────────────────────────────────────────────────

def normalize_token(text: str) -> str:
    """
    Lowercase, strip diacritics, collapse whitespace, remove non-alphanumeric
    characters (except spaces) so that comparison is symbol-agnostic.

    Examples
    --------
    normalize_token("Bicol Express")   → "bicol express"
    normalize_token("Café Pia")        → "cafe pia"
    normalize_token("sili-ice cream")  → "sili ice cream"
    normalize_token("samgyupsal 🔥")   → "samgyupsal"
    """
    if not text:
        return ""
    # Normalize unicode (e.g. accented vowels → base letter)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Lowercase
    text = text.lower()
    # Replace hyphens/underscores with space so "sili-ice" == "sili ice"
    text = re.sub(r"[-_]", " ", text)
    # Remove every remaining non-alphanumeric, non-space character
    text = re.sub(r"[^a-z0-9 ]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Field tokenizer  (handles , and / separators)
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_field(field_value: str) -> list[str]:
    """
    Split a DB field that may contain comma- or slash-separated values into
    a list of normalized tokens.

    Examples
    --------
    tokenize_field("Bicol Express / Laing, Pinangat")
        → ["bicol express", "laing", "pinangat"]

    tokenize_field("Filipino / Korean BBQ")
        → ["filipino", "korean bbq"]

    tokenize_field(None)  → []
    """
    if not field_value:
        return []

    # Split on comma OR slash (with optional surrounding whitespace)
    parts = re.split(r"\s*[,/]\s*", field_value)
    return [normalize_token(p) for p in parts if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Matching helpers
# ─────────────────────────────────────────────────────────────────────────────

def field_matches_query(field_value: str, user_query: str) -> bool:
    """
    Return True if any token from the DB field contains (or equals) the
    normalized user query.

    Supports partial/substring matching so "express" still matches
    "bicol express".

    Parameters
    ----------
    field_value : raw value from the DB column (e.g. "Bicol Express / Laing")
    user_query  : slot value captured by Rasa  (e.g. "bicol express")
    """
    if not field_value or not user_query:
        return False

    needle = normalize_token(user_query)
    haystack_tokens = tokenize_field(field_value)

    return any(needle in token or token in needle for token in haystack_tokens)


def fields_match_all_queries(field_value: str, user_queries: list[str]) -> bool:
    """
    Return True only if EVERY query in user_queries has at least one match
    inside field_value.  Useful for AND-style multi-dish queries.

    Example
    -------
    fields_match_all_queries(
        "Sisig / Crispy Pata, Lechon Kawali",
        ["sisig", "crispy pata"]
    )  →  True
    """
    return all(field_matches_query(field_value, q) for q in user_queries)


# ─────────────────────────────────────────────────────────────────────────────
# SQL LIKE clause builder  (alternative: push matching into the DB layer)
# ─────────────────────────────────────────────────────────────────────────────

def build_like_clauses(column: str, user_queries: list[str]) -> tuple[str, list[str]]:
    """
    Build a SQL WHERE fragment with LIKE conditions for each query token so
    you can do the filtering at the DB level instead of in Python.

    Returns (sql_fragment, params) ready for cursor.execute().

    Example
    -------
    sql, params = build_like_clauses("signature_dish", ["bicol express", "laing"])
    # sql    →  "(LOWER(signature_dish) LIKE ? OR LOWER(signature_dish) LIKE ?)
    #            AND (LOWER(signature_dish) LIKE ? OR LOWER(signature_dish) LIKE ?)"
    # params →  ["%bicol express%", "%bicol express%", "%laing%", "%laing%"]
    
    Note: Each query is matched with LIKE '%query%' so partial matches work.
    The OR within each group handles cases where the token might appear anywhere.
    """
    if not user_queries:
        return ("1=1", [])

    clauses = []
    params = []
    for query in user_queries:
        needle = normalize_token(query)
        # One LIKE per query; covers both comma-separated and slash-separated values
        clauses.append(f"LOWER({column}) LIKE ?")
        params.append(f"%{needle}%")

    return (" AND ".join(clauses), params)


# ─────────────────────────────────────────────────────────────────────────────
# Example integration in an action
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_EXAMPLE = """
# ── In your Rasa action (e.g. action_smart_filter) ──────────────────────────

from db_query_helpers import build_like_clauses, field_matches_query, normalize_token
import sqlite3

def query_by_signature_dish(db_path: str, dish_slot: str | list) -> list[dict]:
    \"\"\"
    dish_slot may be a single string ("bicol express") or a list
    (["bicol express", "laing"]) when the user asked for multiple dishes.
    \"\"\"
    queries = [dish_slot] if isinstance(dish_slot, str) else dish_slot
    queries = [q for q in queries if q]  # drop None/empty

    sql_fragment, params = build_like_clauses("signature_dish", queries)
    sql = f"SELECT * FROM restaurants WHERE {sql_fragment}"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def query_by_cuisine(db_path: str, cuisine_slot: str) -> list[dict]:
    \"\"\"
    cuisine_slot: value from the Rasa cuisine slot ("Korean", "café", etc.)
    Matches against DB rows where the cuisine column may contain
    comma/slash-separated values like "Korean / Japanese".
    \"\"\"
    if not cuisine_slot:
        return []

    sql_fragment, params = build_like_clauses("cuisine", [cuisine_slot])
    sql = f"SELECT * FROM restaurants WHERE {sql_fragment}"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
"""

if __name__ == "__main__":
    # Quick smoke-test
    tests = [
        # (field_value, query, expected)
        ("Bicol Express / Laing, Pinangat", "bicol express", True),
        ("Bicol Express / Laing, Pinangat", "laing", True),
        ("Bicol Express / Laing, Pinangat", "sisig", False),
        ("Filipino / Korean BBQ", "korean bbq", True),
        ("Filipino / Korean BBQ", "korean", True),   # partial
        ("Café / Coffee", "cafe", True),             # diacritic stripped
        ("Sisig, Crispy Pata", "crispy pata", True),
        ("Sisig, Crispy Pata", "sili ice cream", False),
    ]

    passed = 0
    for field, query, expected in tests:
        result = field_matches_query(field, query)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"  {status} field_matches_query({field!r}, {query!r}) = {result} (expected {expected})")

    print(f"\n{passed}/{len(tests)} tests passed")
