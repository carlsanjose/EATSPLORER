"""
Eatsplorer - Custom RASA Actions (Consolidated & Streamlined)

Key fixes:
  - Removed redundant actions (BestOverall, WorstOverall, RestaurantByAspect).
  - SmartFilter now acts as the master handler for all aspect & cuisine queries.
  - restaurant_name slot cleared after all list actions to prevent carryover.
"""

import os
import re
import math
import sqlite3
import pandas as pd
from typing import Any, Text, Dict, List, Optional

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# ─────────────────────────────────────────────
# Text Normalization Helper
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Removes punctuation, extra spaces, and lowercases text for exact matching."""
    if not isinstance(text, str):
        return ""
    # Lowercase the text
    text = text.lower()
    # Remove apostrophes completely (e.g. McDonald's -> mcdonalds)
    text = text.replace("'", "").replace("’", "")
    # Replace all other punctuation and symbols with a single space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─────────────────────────────────────────────
# DB Field Matching Helpers  (db_query_helpers)
# Handles cuisine_type and best_dish_signature_dish fields
# stored with comma (,) and slash (/) separators.
#   e.g. "Filipino / Bicolano"
#   e.g. "Sili Ice Cream, Tinapa Fried Rice"
# ─────────────────────────────────────────────
import unicodedata as _udata

def normalize_token(text: str) -> str:
    """
    Lowercase, strip diacritics (Café→cafe), replace hyphens with spaces,
    remove non-alphanumeric chars.  More thorough than clean_text.
    Used for cuisine / dish matching.
    """
    if not text:
        return ""
    text = _udata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not _udata.combining(c))
    text = text.lower()
    text = re.sub(r"[-_]", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_field(field_value: str) -> list:
    """
    Split a DB field that stores multiple values with , or / separators
    into a list of normalized tokens.
      "Filipino / Bicolano"               → ["filipino", "bicolano"]
      "Sili Ice Cream, Tinapa Fried Rice" → ["sili ice cream", "tinapa fried rice"]
      "Korean BBQ / Hot Pot"              → ["korean bbq", "hot pot"]
    """
    if not field_value:
        return []
    parts = re.split(r"\s*[,/]\s*", str(field_value))
    return [normalize_token(p) for p in parts if p.strip()]

def field_matches_query(field_value: str, user_query: str) -> bool:
    """
    Return True if any token from the DB field contains (or equals) the
    normalized user query.  Supports partial / substring matching so
    "korean"       hits "Korean BBQ / Hot Pot"
    "bicol express" hits "Bicol Express / Laing, Pinangat"
    "cafe"          hits "Coffee Shop / Café"
    """
    if not field_value or not user_query:
        return False
    needle = normalize_token(user_query)
    for token in tokenize_field(field_value):
        if needle in token or token in needle:
            return True
    return False


# ─────────────────────────────────────────────
# Data Loading — SQLite
# ─────────────────────────────────────────────

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "eatsplorer.db")

def load_scores() -> pd.DataFrame:
    """Load restaurant_scores table from SQLite into a DataFrame."""
    conn = sqlite3.connect(_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM restaurant_scores", conn)
    conn.close()
    # NEW: Apply clean_text to every restaurant name
    df["_name_clean"] = df["restaurant_name"].apply(clean_text)
    # Keep this for backward compatibility if other functions use it
    df["_name_lower"] = df["restaurant_name"].str.lower().str.strip() 
    return df

SCORES_DB: pd.DataFrame = load_scores()

def load_scores_monthly() -> pd.DataFrame:
    """Load restaurant_scores_monthly table from SQLite into a DataFrame."""
    conn = sqlite3.connect(_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM restaurant_scores_monthly", conn)
    conn.close()
    df["_name_clean"] = df["restaurant_name"].apply(clean_text)
    df["_name_lower"] = df["restaurant_name"].str.lower().str.strip()
    return df

SCORES_MONTHLY_DB: pd.DataFrame = load_scores_monthly()

def load_info() -> pd.DataFrame:
    """Load restaurant_information table from SQLite into a DataFrame."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM restaurant_information", conn)
    except Exception:
        df = pd.DataFrame(columns=["restaurant_name", "cuisine_type",
                                   "google_maps_address", "best_dish_signature_dish",
                                   "quick_summary"])
    finally:
        conn.close()
    df = df[df["restaurant_name"].notna()].copy()
    
    # NEW: Apply clean_text to every restaurant name
    df["_name_clean"] = df["restaurant_name"].apply(clean_text)
    # Keep this for backward compatibility
    df["_name_lower"] = df["restaurant_name"].str.lower().str.strip()
    
    if "cuisine_type" not in df.columns:
        df["cuisine_type"] = ""
    return df

INFO_DB: pd.DataFrame = load_info()

# ─────────────────────────────────────────────
# Cuisine Detection — keyword-based fallback
# ─────────────────────────────────────────────

NAME_CUISINE_KEYWORDS = {
    "cafe": "Cafe", "café": "Cafe", "coffee": "Coffee Shop", "kape": "Coffee Shop",
    "brew": "Coffee Shop", "tea": "Tea / Cafe", "capsules": "Coffee Shop",
    "korean": "Korean", "samgyup": "Korean BBQ", "kimbap": "Korean",
    "tong yang": "Korean BBQ / Hot Pot", "ramen": "Japanese Ramen",
    "botejyu": "Japanese", "takoyadon": "Japanese", "j.co": "Donuts & Coffee",
    "wasabi": "Japanese", "jollibee": "Filipino Fast Food", "mcdonald": "Fast Food",
    "chowking": "Chinese Fast Food", "mang inasal": "Filipino Grill",
    "greenwich": "Pizza", "angel": "Pizza", "pizza": "Pizza", "ribshack": "BBQ Ribs",
    "starbucks": "Coffee Shop", "figaro": "Coffee Shop", "inasal": "Filipino Grill",
    "bar": "Bar & Restaurant", "grill": "Grill", "seafood": "Seafood",
    "dampa": "Seafood", "sushi": "Japanese", "sumo": "Japanese Wagyu",
    "kyukyu": "Japanese Ramen", "ramenco": "Japanese Ramen", "bigg": "Filipino Diner",
    "diner": "Filipino Diner", "bistro": "Filipino Bistro", "kuya j": "Filipino",
    "lakshmi": "Indian", "indian": "Indian", "hummus": "Middle Eastern",
    "biryani": "Middle Eastern", "wings": "Wings", "pepa": "Wings",
    "yangmatt": "Wings & Milk Tea", "gong cha": "Milk Tea", "snack": "Snack Bar",
    "lounge": "Lounge & Bar", "koki": "Lounge & Bar", "supermarket": "Supermarket",
    "robinson": "Supermarket", "sm super": "Supermarket", "food park": "Food Park",
    "sibid": "Food Park",
}

USER_CUISINE_KEYWORDS = {
    "café": ["cafe", "coffee"], "cafe": ["cafe", "coffee"],
    "coffee": ["coffee", "brew", "kape", "capsules", "figaro", "starbucks"],
    "coffee shop": ["coffee", "brew", "kape", "capsules"],
    "korean": ["korean", "samgyup", "tong yang", "kimbap", "jiah", "seoul", "kim's"],
    "japanese": ["ramen", "botejyu", "takoyadon", "j.co", "wasabi", "sumo", "kyukyu"],
    "ramen": ["ramen", "kyukyu", "ramenco"],
    "chinese": ["hap chan", "chef lee", "ling nam", "chowking", "golden dragon", "four seasons"],
    "filipino": ["jollibee", "mang inasal", "bigg", "kuya j", "cesar", "mamay", "criselda",
                 "kusina", "zoe", "chachi", "inasal", "casa de", "lola feling"],
    "fast food": ["jollibee", "mcdonald", "chowking", "greenwich", "mang inasal"],
    "pizza": ["pizza", "greenwich", "angel's pizza", "s&r"],
    "bbq": ["ribshack", "grill", "smoke", "romantic baboy", "inasal"],
    "seafood": ["dampa", "layag", "seafood", "rockport"],
    "bar": ["alibar", "faris bar", "arang", "koki", "gerry's bar"],
    "wings": ["pepa wings", "yangmatt", "monster wings"],
    "indian": ["lakshmi"],
    "breakfast": ["breakfast republik", "primeroasting"],
    "milk tea": ["gong cha", "island tea", "yangmatt"],
    "grill": ["1st colonial", "gerry's", "mang inasal", "smoke", "ribshack", "romantic baboy"],
    "budget": None, "affordable": None, "cheap": None,
}

def get_cuisine_from_name(restaurant_name: str) -> str:
    name_lower = restaurant_name.lower()
    for keyword, cuisine in NAME_CUISINE_KEYWORDS.items():
        if keyword in name_lower: return cuisine
    return "Restaurant"

def get_cuisine(restaurant_name: str) -> str:
    if not INFO_DB.empty:
        q = restaurant_name.lower().strip()
        match = INFO_DB[INFO_DB["_name_lower"] == q]
        if match.empty:
            match = INFO_DB[INFO_DB["_name_lower"].str.contains(q, na=False, regex=False)]
        if not match.empty:
            val = str(match.iloc[0].get("cuisine_type", "")).strip()
            if val and val not in ("", "nan", "None"): return val
    return get_cuisine_from_name(restaurant_name)

def filter_by_cuisine_slot(df: pd.DataFrame, cuisine: Any):
    if not cuisine:
        return df, "Legazpi City"
        
    # --- NEW FIX: If Rasa sends a list, just grab the first cuisine ---
    if isinstance(cuisine, list):
        if len(cuisine) > 0:
            cuisine = cuisine[0]
        else:
            return df, "Legazpi City"
            
    # Safety check just in case
    if not isinstance(cuisine, str):
         return df, "Legazpi City"
    # -----------------------------------------------------------------
        
    # Standardize the search term
    q = cuisine.lower().strip()
    
    # Assuming INFO_DB has a 'cuisine_type' column you can filter against
    if not INFO_DB.empty and "cuisine_type" in INFO_DB.columns:
        # Find all restaurants that match the requested cuisine
        # Use field_matches_query so "korean" matches "Korean BBQ / Hot Pot",
        # "cafe" matches "Coffee Shop / Café", etc. (handles , and / separators)
        matched_info = INFO_DB[INFO_DB["cuisine_type"].apply(
            lambda v: field_matches_query(v, cuisine)
        )]
        matched_names = matched_info["restaurant_name"].tolist()
        
        if matched_names:
            # Filter the main SCORES_DB to only include those matched restaurants
            return df[df["restaurant_name"].isin(matched_names)], cuisine.title()
            
    # Fallback: Just look for the cuisine word in the restaurant's name
    return df[df["_name_lower"].str.contains(q, na=False)], cuisine.title()

# ─────────────────────────────────────────────
# Core Helpers
# ─────────────────────────────────────────────

ASPECT_MAP = {
    "food_quality": ("food_quality_avg","food_quality_polarity","food_quality_review_count","🍴 Food Quality"),
    "service":      ("service_avg","service_polarity","service_review_count","🛎️ Service"),
    "ambiance":     ("ambiance_avg","ambiance_polarity","ambiance_review_count","🌿 Ambiance"),
    "price_value":  ("price_value_avg","price_value_polarity","price_value_review_count","💰 Price/Value"),
    "overall":      ("overall_avg","overall_polarity","overall_review_count","⭐ Overall Sentiment"),
}
POLARITY_EMOJI = {"Positive":"✅","Neutral":"🟡","Negative":"❌","N/A":"⬜"}
DEFAULT_TOP_N  = 5

def normalize_aspect(raw: str) -> Optional[str]:
    if not raw: return None
    raw = raw.lower().strip()
    mappings = {
        "food_quality": ["food quality","food","cuisine","dishes","taste","flavors","the food","meal","meals","eat"],
        "service":      ["service","staff","customer service","hospitality","waiters","waiter","servers","crew"],
        "ambiance":     ["ambiance","atmosphere","vibe","decor","environment","setting","place","ambience","feel","mood","ambient","atmosphere"],
        "price_value":  ["price value","price-to-value","value for money","value","price","affordability","budget","cost","cheap","affordable","inexpensive","budget-friendly"],
        "overall":      ["overall","general","sentiment","general sentiment","overall sentiment","general feeling","impression","overall impression","overall rating"],
    }
    for key, synonyms in mappings.items():
        if raw == key or raw in synonyms: return key
    return None

def extract_number_from_text(text: str) -> Optional[int]:
    m = re.search(r'\b(top|give me|show|list|best)?\s*(\d+)\b', text.lower())
    if m:
        n = int(m.group(2))
        if 1 <= n <= 20: return n
    words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    for word, val in words.items():
        if word in text.lower(): return val
    return None

def safe_float(val) -> Optional[float]:
    try:
        if val is None: return None
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError): return None

def format_score(avg, polarity, count) -> str:
    if pd.isna(avg) or count == 0: return "⬜ N/A"
    emoji = POLARITY_EMOJI.get(polarity, "⬜")
    return f"{emoji} {avg:.2f}/5.00  ({int(count)} reviews)"

def format_restaurant_block(row: pd.Series, rank: int = None, highlight_aspect: str = None) -> str:
    name = row["restaurant_name"]
    overall = safe_float(row.get("composite_score"))
    pol = str(row.get("composite_polarity",""))
    total = int(row["total_reviews"]) if not pd.isna(row.get("total_reviews",float("nan"))) else 0
    cuisine = get_cuisine(name)

    prefix = f"**#{rank}** " if rank else ""
    lines  = [f"{prefix}**{name}**"]
    if cuisine: lines.append(f"*{cuisine}*")
    lines.append(f"Overall: {POLARITY_EMOJI.get(pol,'⬜')} {overall:.2f}/5.00  ({total} reviews)" if overall else "Overall: N/A")
    if highlight_aspect and highlight_aspect in ASPECT_MAP:
        avg_col, pol_col, cnt_col, label = ASPECT_MAP[highlight_aspect]
        lines.append(f"{label}: {format_score(row[avg_col], row[pol_col], row[cnt_col])}")
    return "\n".join(lines)

def fuzzy_match_restaurant(query: str, df: pd.DataFrame):
    if not query: return None
    
    # CLEAN THE USER'S QUERY!
    q = clean_text(query)
    
    # 1. Try an exact match on the fully cleaned text
    exact = df[df["_name_clean"] == q]
    if not exact.empty: return exact.iloc[0]
    
    # 2. Try a partial match on the cleaned text
    sub = df[df["_name_clean"].str.contains(q, na=False, regex=False)]
    if not sub.empty: return sub.iloc[0]
    
    # 3. Try matching individual words longer than 3 characters
    words = [w for w in q.split() if len(w) > 3]
    if words:
        temp_match = df.copy()
        for word in words:
            temp_match = temp_match[temp_match["_name_clean"].str.contains(word, na=False, regex=False)]
        if not temp_match.empty: return temp_match.iloc[0]
        
    return None

def get_top_n(df: pd.DataFrame, sort_col: str, n: int, min_reviews_col: str = None) -> pd.DataFrame:
    filtered = df[df[sort_col].notna()]
    if min_reviews_col: filtered = filtered[filtered[min_reviews_col] >= 1]
    return filtered.sort_values(sort_col, ascending=False).head(n)

def get_worst_n(df: pd.DataFrame, sort_col: str, n: int, min_reviews_col: str = None) -> pd.DataFrame:
    filtered = df[df[sort_col].notna()]
    if min_reviews_col: filtered = filtered[filtered[min_reviews_col] >= 1]
    return filtered.sort_values(sort_col, ascending=True).head(n)

def build_list_response(header: str, rows, aspect_key: str = None, tip: str = None) -> str:
    parts = [header, ""]
    for i, (_, row) in enumerate(rows.iterrows(), 1):
        parts.append(format_restaurant_block(row, rank=i, highlight_aspect=aspect_key))
        parts.append("")
    if tip: parts.append(tip)
    return "\n".join(parts).strip()

def row_to_card(row: pd.Series) -> dict:
    aspects_scored = 0
    card = {
        "restaurant_name": row["restaurant_name"],
        "composite_score":   safe_float(row.get("composite_score")),
        "composite_polarity": str(row.get("composite_polarity", "")) or None,
        "total_reviews":   int(row["total_reviews"]) if not pd.isna(row.get("total_reviews", float("nan"))) else 0,
    }
    for key, (avg_col, pol_col, cnt_col, _) in ASPECT_MAP.items():
        avg = safe_float(row.get(avg_col))
        cnt = int(row[cnt_col]) if not pd.isna(row.get(cnt_col, float("nan"))) else 0
        if avg is not None and cnt > 0:
            aspects_scored += 1
            card[key] = {"avg": round(avg, 4), "polarity": str(row.get(pol_col, "")), "review_count": cnt}
        else:
            card[key] = {"avg": None, "polarity": "N/A", "review_count": 0}
    card["aspects_scored"] = aspects_scored
    return card

def dispatch_cards(dispatcher: CollectingDispatcher, rows: pd.DataFrame) -> None:
    if rows.empty: return
    dispatcher.utter_message(json_message={
        "type": "restaurant_cards",
        "restaurants": [row_to_card(row) for _, row in rows.iterrows()],
    })

def get_entity_from_message(tracker: Tracker, entity_name: str) -> Optional[str]:
    for ent in tracker.latest_message.get("entities", []):
        if ent.get("entity") == entity_name: return ent.get("value")
    return None

RESET_SLOTS = [SlotSet("restaurant_name", None)]


# ─────────────────────────────────────────────
# ACTION: Top Restaurants (Overall)
# ─────────────────────────────────────────────

class ActionRecommendTopRestaurants(Action):
    def name(self): return "action_recommend_top_restaurants"
    def run(self, dispatcher, tracker, domain):
        top = get_top_n(SCORES_DB, "composite_score", DEFAULT_TOP_N, "total_reviews")
        if top.empty:
            dispatcher.utter_message(text="Sorry, I couldn't find restaurant data right now.")
            return RESET_SLOTS
        dispatcher.utter_message(text=build_list_response(
            header=f"Sure! Here are the top {len(top)} restaurants in Legazpi City based on overall ratings — tap any card for the full breakdown! 🍽️",
            rows=top,
            tip="💡 *Ask me about a specific restaurant, or filter by food quality, service, ambiance, or value!*"
        ))
        dispatch_cards(dispatcher, top)
        return RESET_SLOTS


# ─────────────────────────────────────────────
# ACTION: Smart Filter (Master Handler for Aspects/Keywords)
# ─────────────────────────────────────────────

class ActionSmartFilter(Action):
    def name(self): return "action_smart_filter"
    def run(self, dispatcher, tracker, domain):
        msg_text = tracker.latest_message.get("text","")
        entities = tracker.latest_message.get("entities",[])

        # 1. Grab slots as potential lists (Ultimate Smart Filter logic)
        raw_cuisines = tracker.get_slot("cuisine")
        raw_aspects = tracker.get_slot("aspect")
        raw_dishes = tracker.get_slot("signature_dish")

        cuisines = [raw_cuisines] if isinstance(raw_cuisines, str) else (raw_cuisines or [])
        aspects = [raw_aspects] if isinstance(raw_aspects, str) else (raw_aspects or [])
        dishes = [raw_dishes] if isinstance(raw_dishes, str) else (raw_dishes or [])

        # 2. Filter by MULTIPLE Cuisines (Looping through your original function!)
        df = SCORES_DB.copy()
        cuisine_labels = []
        if cuisines:
            for c in cuisines:
                df, c_label = filter_by_cuisine_slot(df, c)
                if c_label != "Legazpi City" and c_label not in cuisine_labels:
                    cuisine_labels.append(c_label)

        # 3. Filter by MULTIPLE Signature Dishes
        dish_labels = []
        if dishes:
            info = INFO_DB.copy()
            # Use field_matches_query so "bicol express" matches
            # "Bicol Express / Laing, Pinangat" (handles , and / separators,
            # diacritics, and partial substrings)
            matched_info = info[info["best_dish_signature_dish"].apply(
                lambda v: any(field_matches_query(v, d) for d in dishes)
            )]
            df = df[df['restaurant_name'].isin(matched_info['restaurant_name'])]
            dish_labels = [d.title() for d in dishes]

        # 4. Grab Aspects (Combining entities, slots, and budget keywords)
        aspect_keys = []
        for ent in entities:
            if ent["entity"] == "aspect":
                key = normalize_aspect(ent["value"])
                if key and key not in aspect_keys: aspect_keys.append(key)
        
        for asp in aspects:
            key = normalize_aspect(asp)
            if key and key not in aspect_keys: aspect_keys.append(key)

        msg_lower = msg_text.lower()
        budget_words = ["budget","affordable","cheap","inexpensive","budget-friendly"]
        if any(w in msg_lower for w in budget_words) and "price_value" not in aspect_keys:
            aspect_keys.append("price_value")

        # 5. Apply the Sorting & Scoring (Your EXACT original logic)
        if len(aspect_keys) >= 2:
            score_cols = [ASPECT_MAP[k][0] for k in aspect_keys]
            df_v = df.dropna(subset=score_cols).copy()
            if not df_v.empty:
                df_v["_score"] = df_v[score_cols].mean(axis=1)
                top = df_v.sort_values("_score", ascending=False).head(DEFAULT_TOP_N)
            else: 
                top = pd.DataFrame()
            sort_label = " + ".join(ASPECT_MAP[k][3] for k in aspect_keys)
        elif aspect_keys:
            avg_col = ASPECT_MAP[aspect_keys[0]][0]
            cnt_col = ASPECT_MAP[aspect_keys[0]][2]
            df_v = df[df[avg_col].notna() & df[cnt_col].ge(1)]
            top = df_v.sort_values(avg_col, ascending=False).head(DEFAULT_TOP_N)
            sort_label = ASPECT_MAP[aspect_keys[0]][3]
        else:
            df_v = df[df["composite_score"].notna() & df["total_reviews"].ge(1)]
            top = df_v.sort_values("composite_score", ascending=False).head(DEFAULT_TOP_N)
            sort_label = "overall score"

        # 6. Fallback if no matching restaurants found
        if top.empty:
            fallback = get_top_n(SCORES_DB, "composite_score", DEFAULT_TOP_N, "total_reviews")
            dispatcher.utter_message(text=(
                "I couldn't find a perfect match for that combination in our database, "
                f"but here are some highly rated restaurants you might enjoy:\n\n"
                + build_list_response("", rows=fallback, tip="💡 *Try asking with fewer filters for more results!*")
            ))
            return RESET_SLOTS

        # 7. Format the Custom Header
        header_parts = []
        filter_labels = cuisine_labels + dish_labels
        
        if filter_labels:
            header_parts.append(" & ".join(filter_labels))
            
        if aspect_keys: 
            header_parts.append(f"best {sort_label}")
            
        header = "Here's what I found for " + (" · ".join(header_parts) if header_parts else "best matches")

        # 8. Output Text & React Cards (Your EXACT original dispatch methods)
        dispatcher.utter_message(text=build_list_response(
            header=header, rows=top, aspect_key=aspect_keys[0] if aspect_keys else None,
            tip="💡 *Ask me for more details on any of these restaurants!*"
        ))
        
        dispatch_cards(dispatcher, top)
        
        # 9. Clear all slots
        return [
            SlotSet("aspect", aspect_keys[0] if aspect_keys else None), 
            SlotSet("restaurant_name", None), 
            SlotSet("cuisine", None),
            SlotSet("signature_dish", None)
        ]


# ─────────────────────────────────────────────
# ACTION: Top N by Aspect
# ─────────────────────────────────────────────

class ActionTopNByAspect(Action):
    def name(self): return "action_top_n_by_aspect"
    def run(self, dispatcher, tracker, domain):
        n_raw = tracker.get_slot("number")
        if n_raw: n = max(1, min(int(float(n_raw)), 20))
        else:
            extracted = extract_number_from_text(tracker.latest_message.get("text", ""))
            n = extracted if extracted else DEFAULT_TOP_N

        raw = get_entity_from_message(tracker, "aspect") or tracker.get_slot("aspect")
        aspect_key = normalize_aspect(raw)

        if aspect_key:
            avg_col,_,cnt_col,label = ASPECT_MAP[aspect_key]
            top = get_top_n(SCORES_DB, avg_col, n, cnt_col)
            header = f"**Of course! Here are the top {n} Restaurants ranked by:** {label} 🍽️"
        else:
            top = get_top_n(SCORES_DB, "composite_score", n, "total_reviews")
            header = f"🍽️ **Top {n} Restaurants in Legazpi City** (Overall)"

        if top.empty:
            dispatcher.utter_message(text="Sorry, I couldn't find restaurant data for that query.")
            return RESET_SLOTS

        dispatcher.utter_message(text=build_list_response(header=header, rows=top, aspect_key=aspect_key))
        dispatch_cards(dispatcher, top)
        return [SlotSet("number", None), SlotSet("restaurant_name", None)]





# ─────────────────────────────────────────────
# ACTION: Restaurant Info
# ─────────────────────────────────────────────

class ActionRestaurantInfo(Action):
    def name(self): return "action_restaurant_info"
    def run(self, dispatcher, tracker, domain):
        restaurant_name = get_entity_from_message(tracker, "restaurant_name")
        if not restaurant_name: restaurant_name = tracker.get_slot("restaurant_name")

        if not restaurant_name:
            dispatcher.utter_message(text="Which restaurant? Please mention the name!")
            return []

        row = fuzzy_match_restaurant(restaurant_name, SCORES_DB)

        if row is None:
            top3 = get_top_n(SCORES_DB, "composite_score", 3, "total_reviews")
            suggestions = ", ".join(f"**{r['restaurant_name']}**" for _,r in top3.iterrows())
            dispatcher.utter_message(
                text=(f"I couldn't find **'{restaurant_name}'** in my database. "
                      f"Here are some top-rated restaurants you can ask about:\n{suggestions}")
            )
            return [SlotSet("restaurant_name", None)]

        name = row["restaurant_name"]
        overall = safe_float(row.get("composite_score"))
        pol = str(row.get("composite_polarity",""))
        total = int(row["total_reviews"]) if not pd.isna(row.get("total_reviews",float("nan"))) else 0
        cuisine = get_cuisine(name)

        asked_aspect = get_entity_from_message(tracker, "aspect")
        if not asked_aspect: asked_aspect = normalize_aspect(tracker.get_slot("aspect") or "")
        
        if not asked_aspect:
            msg_lower = tracker.latest_message.get("text","").lower()
            aspect_hints = {
                "food_quality": ["food","meal","dish","taste","cuisine","eat"],
                "service":      ["service","staff","waiter","crew"],
                "ambiance":     ["ambiance","atmosphere","vibe","decor","feel","ambient"],
                "price_value":  ["price","cost","budget","value","affordable","cheap"],
            }
            for key, hints in aspect_hints.items():
                if any(h in msg_lower for h in hints):
                    asked_aspect = key; break

        overall_str = f"{POLARITY_EMOJI.get(pol,'⬜')} {overall:.2f}/5.00  ({total} reviews)" if overall is not None else "N/A"
        lines = [f"Sure! Here's what our review data says about **{name}** 📍"]
        if cuisine: lines.append(f"*{cuisine}*")
        lines += ["", f"⭐ **Overall Score:** {overall_str}", ""]

        dispatcher.utter_message(text="\n".join(lines))
        dispatch_cards(dispatcher, SCORES_DB[SCORES_DB["restaurant_name"] == name].head(1))
        return [SlotSet("restaurant_name", name), SlotSet("aspect", None)]


# ─────────────────────────────────────────────
# ACTION: Positive Only
# ─────────────────────────────────────────────

class ActionPositiveOnly(Action):
    def name(self): return "action_positive_only"
    def run(self, dispatcher, tracker, domain):
        positive = SCORES_DB[SCORES_DB["composite_polarity"] == "Positive"]
        top = positive.sort_values("composite_score", ascending=False).head(DEFAULT_TOP_N)
        if top.empty:
            dispatcher.utter_message(text="No positively reviewed restaurants found.")
            return RESET_SLOTS
        dispatcher.utter_message(text=build_list_response(
            header=f"Great news! Here are {len(top)} Restaurants with glowing reviews from customers ✅", rows=top
        ))
        dispatch_cards(dispatcher, top)
        return RESET_SLOTS


# ─────────────────────────────────────────────
# ACTION: Negative Warning
# ─────────────────────────────────────────────

class ActionNegativeWarning(Action):
    def name(self): return "action_negative_warning"
    def run(self, dispatcher, tracker, domain):
        # 1. Dynamically figure out the number (n)
        n_raw = tracker.get_slot("number")
        if n_raw: 
            n = max(1, min(int(float(n_raw)), 20))
        else:
            extracted = extract_number_from_text(tracker.latest_message.get("text", ""))
            n = extracted if extracted else DEFAULT_TOP_N

        # 2. Figure out the aspect
        raw = get_entity_from_message(tracker, "aspect") or tracker.get_slot("aspect")
        aspect_key = normalize_aspect(raw)
        
        # 3. Filter and sort the database
        if aspect_key and aspect_key in ASPECT_MAP:
            avg_col, pol_col, cnt_col, label = ASPECT_MAP[aspect_key]
            # Look for negative polarity first
            negative = SCORES_DB[SCORES_DB[pol_col] == "Negative"]
            if negative.empty:
                # Fallback: Just grab the lowest scores for that aspect
                negative = SCORES_DB[SCORES_DB[avg_col].notna() & SCORES_DB[cnt_col].ge(1)]
            sort_col = avg_col
            header = f"Noted! Just so you know, these restaurants have received negative feedback on {label} — worth considering before visiting these {n} Restaurants ⚠️"
        else:
            negative = SCORES_DB[SCORES_DB["composite_polarity"] == "Negative"]
            if negative.empty:
                negative = SCORES_DB[SCORES_DB["composite_score"].notna() & SCORES_DB["total_reviews"].ge(1)]
            sort_col = "composite_score"
            header = f"Worth considering before visiting these {n} Restaurants ⚠️"
            
        if negative.empty:
            dispatcher.utter_message(text="Good news! No low-rated restaurants found. 🎉")
            return RESET_SLOTS
            
        # 4. Sort ascending (lowest first) and format
        neg_sorted = negative.sort_values(sort_col, ascending=True).head(n)
        
        parts = [header, "*(Consider these carefully based on customer feedback)*", ""]
        
            
        dispatcher.utter_message(text="\n".join(parts).strip())
        dispatch_cards(dispatcher, neg_sorted)
        
        # Reset slots so they don't carry over
        return [SlotSet("number", None), SlotSet("restaurant_name", None), SlotSet("aspect", None)]

# ─────────────────────────────────────────────
# ACTION: Live Review + ABSA Inference
# ─────────────────────────────────────────────
 
class ActionLiveReviewSummary(Action):
    def name(self): return "action_live_review_summary"
    def run(self, dispatcher, tracker, domain):
        restaurant_name = get_entity_from_message(tracker, "restaurant_name") or tracker.get_slot("restaurant_name")
        if not restaurant_name:
            dispatcher.utter_message(text="Which restaurant would you like the latest review for?")
            return []
            
        import requests as req
        fastapi_url = os.getenv("FASTAPI_URL","http://localhost:8000")
        try:
            resp = req.post(f"{fastapi_url}/api/live-reviews", json={"restaurant_name":restaurant_name}, timeout=200)
            if resp.status_code == 200:
                data = resp.json()
                summary = data.get("summary_text", "")
                reviews = data.get("reviews", [])
                lines = [summary, ""]
                if reviews:
                    # Show up to 5 reviews with their ABSA analysis
                    aspect_labels = {
                        "overall":      "Overall",
                        "food_quality": "Food Quality",
                        "service":      "Service",
                        "ambiance":     "Ambiance",
                        "price_value":  "Price / Value",
                    }
                    pol_emoji = {"Positive": "✅", "Neutral": "🟡", "Negative": "❌"}
                    top_reviews = reviews[:5]
                    lines.append(f"**📝 Latest {len(top_reviews)} Google Review{'s' if len(top_reviews) != 1 else ''}:**")
                    lines.append("")
                    for i, review in enumerate(top_reviews, start=1):
                        text   = review.get("text", "")
                        author = review.get("author", "Anonymous")
                        rating = review.get("rating")
                        time   = review.get("time", "")
                        stars  = f"{rating}/5 ⭐" if rating else "unrated"
                        lines.append(f"**Review {i}**")
                        lines += [f'*"{text}"*', f"— {author} · {stars} · {time}"]
                        absa = review.get("absa_inference")
                        if absa:
                            lines += ["", "**Eatsplorer Analysis:**"]
                            for key, label in aspect_labels.items():
                                result    = absa.get(key, {})
                                sentiment = result.get("sentiment", "N/A")
                                score     = result.get("score")
                                score_str = f" ({score:.2f})" if score is not None else ""
                                icon      = pol_emoji.get(sentiment, "⬜")
                                lines.append(f"  {label}: {icon} {sentiment}{score_str}")
                        lines.append("")  # blank line between reviews
                else:
                    lines.append("No recent text reviews found on Google.")
                dispatcher.utter_message(text="\n".join(lines))
            elif resp.status_code == 503:
                dispatcher.utter_message(text="Live review lookup needs a Google Places API key.")
            elif resp.status_code == 404:
                dispatcher.utter_message(text=f"Couldn't find **{restaurant_name}** on Google Places.")
            else:
                dispatcher.utter_message(text="Couldn't fetch a live review right now. Try again shortly!")
        except Exception as e:
            dispatcher.utter_message(text="Live review lookup temporarily unavailable.")
        return [SlotSet("restaurant_name", restaurant_name)]


# ─────────────────────────────────────────────
# ACTION: Location Lookup
# ─────────────────────────────────────────────

class ActionLocation(Action):
    def name(self): return "action_location"
    def run(self, dispatcher, tracker, domain):
        # Always prefer the entity from the CURRENT message — never rely solely on slot
        restaurant_name = get_entity_from_message(tracker, "restaurant_name")
        if not restaurant_name:
            restaurant_name = tracker.get_slot("restaurant_name")

        print(f"DEBUG: Rasa extracted restaurant_name slot as: '{tracker.get_slot('restaurant_name')}'")

        if not restaurant_name:
            dispatcher.utter_message(text="Which restaurant would you like to find?")
            return [SlotSet("restaurant_name", None)]

        name, address, sig_dish, summary = restaurant_name, None, None, None
        cuisine = get_cuisine(restaurant_name)

        if not INFO_DB.empty:
            q = restaurant_name.lower().strip()
            match = INFO_DB[INFO_DB["_name_lower"] == q]
            if match.empty:
                match = INFO_DB[INFO_DB["_name_lower"].str.contains(q, na=False, regex=False)]
            if match.empty:
                # Only use words longer than 3 chars to avoid false partial matches
                words = [w for w in q.split() if len(w) > 3]
                if words:
                    temp_match = INFO_DB.copy()
                    # Force it to match ALL words, not just the first one
                    for word in words:
                        temp_match = temp_match[temp_match["_name_lower"].str.contains(word, na=False, regex=False)]
                    if not temp_match.empty:
                        match = temp_match

        if not match.empty:
            row = match.iloc[0]
            name    = row.get("restaurant_name", restaurant_name)
            raw_addr = row.get("google_maps_address")
            address  = str(raw_addr).strip() if raw_addr and not isinstance(raw_addr, float) and str(raw_addr).strip() not in ("", "nan") else None
            raw_dish = row.get("best_dish_signature_dish")
            sig_dish = str(raw_dish).strip() if raw_dish and not isinstance(raw_dish, float) and str(raw_dish).strip() not in ("", "nan") else None
            raw_sum  = row.get("quick_summary")
            summary  = str(raw_sum).strip() if raw_sum and not isinstance(raw_sum, float) and str(raw_sum).strip() not in ("", "nan") else None
            ct = row.get("cuisine_type", "")
            if ct and not isinstance(ct, float) and str(ct).strip() not in ("", "nan"):
                cuisine = str(ct).strip()

        # Build proper Google Maps URLs (not raw address text)
        dest_raw = address if address else f"{name} Legazpi City Philippines"
        dest_enc = dest_raw.replace(" ", "+")
        nav_url  = f"https://www.google.com/maps/dir/?api=1&destination={dest_enc}"
        maps_url = f"https://www.google.com/maps/search/?api=1&query={dest_enc}"

        lines = [f"📍 **{name}**"]
        if cuisine: lines.append(f"*{cuisine}*")
        lines.append("")
        if address:
            lines.append(f"**Address:** {address}")
            lines.append(f"🗺️ **Google Maps:** {maps_url}")
        else:
            lines.append("Address not yet in our database.")
            lines.append(f"🗺️ **Search on Google Maps:** {maps_url}")
        lines.append(f"🧭 **Get Directions:** {nav_url}")
        if sig_dish: lines.append(f"\n🍽️ **Signature Dish:** {sig_dish}")
        if summary:  lines.append(f"\n*{summary}*")

        # --- UPDATED: Send BOTH the Markdown text and the custom map payload ---
        dispatcher.utter_message(
            text="\n".join(lines),
            custom={
                "payload": "inline_map",
                "restaurant_name": name,
                "address": dest_raw
            }
        )
        
        # Clear slot so next location query starts fresh
        return [SlotSet("restaurant_name", None)]


# ─────────────────────────────────────────────
# ACTION: Culinary Itinerary Builder
# ─────────────────────────────────────────────

TIME_SLOTS = [
    ("7:00 AM",  "Breakfast"), ("10:00 AM", "Morning Break"),
    ("12:00 PM", "Lunch"), ("3:00 PM",  "Afternoon Snack"),
    ("6:00 PM",  "Dinner"), ("8:00 PM",  "Dessert / After-dinner"),
]

class ActionItineraryBuilder(Action):
    def name(self): return "action_itinerary_builder"
    def run(self, dispatcher, tracker, domain):
        msg_text = tracker.latest_message.get("text", "")

    
        n_raw = tracker.get_slot("number")
        if n_raw: n_stops = max(2, min(int(float(n_raw)), 6))
        else:
            extracted = extract_number_from_text(msg_text)
            n_stops = extracted if extracted else 5

        cuisine_requested = tracker.get_slot("cuisine")

        if cuisine_requested:
            df, cuisine_label = filter_by_cuisine_slot(SCORES_DB, cuisine_requested)
        else:
            df, cuisine_label = SCORES_DB, "Legazpi City"

        df_valid = df[df["composite_score"].notna() & df["total_reviews"].ge(1)]
        top = df_valid.sort_values("composite_score", ascending=False).head(n_stops)

        if top.empty:
            dispatcher.utter_message(text=f"I couldn't find enough restaurants for a {cuisine_label} itinerary. Try a broader category!")
            return [SlotSet("restaurant_name", None), SlotSet("number", None)]

        lines = [f"🗓️ **Your {cuisine_label} Itinerary for a Day in Legazpi City**", ""]
        for i, (_, row) in enumerate(top.iterrows()):
            if i >= len(TIME_SLOTS): break
            time_str, meal_label = TIME_SLOTS[i]
            name = row["restaurant_name"]
            overall = safe_float(row.get("composite_score"))
            cuisine = get_cuisine(name)
            
            lines.append(f"**{time_str} — {meal_label}**")
            lines.append(f"📍 **{name}**")
            if cuisine: lines.append(f"*{cuisine}*")
            lines.append(f"Rating: ✅ {overall:.2f}/5.00" if overall else "Rating: ✅ N/A")
            lines.append(f"🧭 Directions: https://www.google.com/maps/dir/?api=1&destination={name.replace(' ', '+')}+Legazpi+City+Philippines\n")

        dispatcher.utter_message(text="\n".join(lines))
        return [SlotSet("restaurant_name", None), SlotSet("number", None), SlotSet("cuisine", None)]


# ─────────────────────────────────────────────
# ACTION: Best Restaurant by Month
# ─────────────────────────────────────────────

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}
MONTH_NAME_TO_NUM = {v.lower(): k for k, v in MONTH_NAMES.items()}
MONTH_ABBR_TO_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def parse_month_from_text(text: str) -> Optional[int]:
    """Extract a month number from a user message. Handles full names, abbreviations, and numbers."""
    text_lower = text.lower()
    for name, num in MONTH_NAME_TO_NUM.items():
        if name in text_lower:
            return num
    for abbr, num in MONTH_ABBR_TO_NUM.items():
        if abbr in text_lower:
            return num
    m = re.search(r'\bmonth\s*(\d{1,2})\b|\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*month\b', text_lower)
    if m:
        num = int(m.group(1) or m.group(2))
        if 1 <= num <= 12:
            return num
    return None

def parse_year_from_text(text: str) -> Optional[int]:
    """Extract a 4-digit year from a user message."""
    m = re.search(r'\b(202[4-9]|203\d)\b', text)
    return int(m.group(1)) if m else None


class ActionBestByMonth(Action):
    def name(self): return "action_best_by_month"

    def run(self, dispatcher, tracker, domain):
        import datetime
        msg_text = tracker.latest_message.get("text", "")

        # 1. Parse month and year from message
        month = parse_month_from_text(msg_text)
        year  = parse_year_from_text(msg_text)

        # Fallback: use previous month if neither is specified
        if not month:
            today = datetime.date.today()
            first_of_this_month = today.replace(day=1)
            prev = first_of_this_month - datetime.timedelta(days=1)
            month = prev.month
            year  = year or prev.year
        if not year:
            year = datetime.date.today().year

        month_name = MONTH_NAMES.get(month, str(month))

        # 2. How many results?
        n_raw = tracker.get_slot("number")
        if n_raw:
            n = max(1, min(int(float(n_raw)), 20))
        else:
            extracted = extract_number_from_text(msg_text)
            n = extracted if extracted else DEFAULT_TOP_N

        # 3. Optional cuisine filter
        cuisine_slot = tracker.get_slot("cuisine")

        # 4. Filter monthly table for the requested month/year
        df = SCORES_MONTHLY_DB.copy()
        df_month = df[(df["Year"] == year) & (df["Month"] == month)]

        if df_month.empty:
            dispatcher.utter_message(
                text=f"I don't have enough data for **{month_name} {year}** yet. "
                     f"Try asking about a different month!"
            )
            return [SlotSet("number", None), SlotSet("cuisine", None), SlotSet("restaurant_name", None)]

        # 5. Apply cuisine filter if provided
        if cuisine_slot:
            if not INFO_DB.empty and "cuisine_type" in INFO_DB.columns:
                matched_info = INFO_DB[INFO_DB["cuisine_type"].apply(
                    lambda v: field_matches_query(v, cuisine_slot)
                )]
                matched_names = matched_info["restaurant_name"].tolist()
                df_month = df_month[df_month["restaurant_name"].isin(matched_names)]

        # 6. Support aspect filtering (default to composite_score for overall ranking)
        aspect_raw = tracker.get_slot("aspect")
        aspect_key = normalize_aspect(aspect_raw)
        if aspect_key and aspect_key in ASPECT_MAP:
            avg_col, pol_col, cnt_col, aspect_label = ASPECT_MAP[aspect_key]
            df_valid = df_month[df_month[avg_col].notna() & df_month[cnt_col].ge(1)]
            top = df_valid.sort_values(avg_col, ascending=False).head(n)
        else:
            # Use composite_score (weighted aggregate) as the primary sort — same as
            # the overall rankings — to avoid many ties that occur with overall_avg alone.
            avg_col, pol_col, cnt_col, aspect_label = ASPECT_MAP["overall"]
            df_valid = df_month[
                df_month["composite_score"].notna() & df_month["total_reviews"].ge(1)
            ]
            top = df_valid.sort_values("composite_score", ascending=False).head(n)

        if top.empty:
            dispatcher.utter_message(
                text=f"No restaurants with enough reviews found for **{month_name} {year}**."
                     + (f" Try removing the cuisine filter!" if cuisine_slot else "")
            )
            return [SlotSet("number", None), SlotSet("cuisine", None), SlotSet("restaurant_name", None)]

        # 7. Build header
        cuisine_label = cuisine_slot.title() if cuisine_slot else None
        aspect_label_str = f" ({aspect_label})" if aspect_key else ""
        header_parts = []
        if cuisine_label:
            header_parts.append(cuisine_label)
        header = (
            f"🏆 **Top {len(top)} "
            + (" ".join(header_parts) + " " if header_parts else "")
            + f"Restaurants for {month_name} {year}{aspect_label_str}** — based on reviews from that month:"
        )

        # 8. Build text list (reuse format_restaurant_block via aspect mapping)
        parts = [header, ""]
        for i, (_, row) in enumerate(top.iterrows(), 1):
            name     = row["restaurant_name"]
            avg      = safe_float(row.get(avg_col))
            pol      = str(row.get(pol_col, ""))
            total    = int(row[cnt_col]) if not pd.isna(row.get(cnt_col, float("nan"))) else 0
            cuisine  = get_cuisine(name)

            parts.append(f"**#{i}** **{name}**")
            if cuisine:
                parts.append(f"*{cuisine}*")
            if avg is not None:
                parts.append(
                    f"{aspect_label}: {POLARITY_EMOJI.get(pol, '⬜')} {avg:.2f}/5.00  ({total} reviews)"
                )
            parts.append("")

        parts.append("💡 *Ask me about a specific restaurant or try another month!*")
        dispatcher.utter_message(text="\n".join(parts).strip())

        # 9. Dispatch cards — use monthly scores directly so numbers reflect that month,
        #    not the all-time averages from SCORES_DB.
        if not top.empty:
            dispatcher.utter_message(json_message={
                "type": "restaurant_cards",
                "restaurants": [row_to_card(row) for _, row in top.iterrows()],
                "aspect": aspect_key or None
            })

        return [
            SlotSet("number", None),
            SlotSet("cuisine", None),
            SlotSet("restaurant_name", None),
        ]