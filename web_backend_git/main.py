"""
Eatsplorer FastAPI Backend
- Proxies chat messages to the RASA REST webhook
- Serves restaurant data directly from restaurant_scores.csv
- CORS enabled for the React frontend
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import pandas as pd
import sqlite3
import math
import os


# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Eatsplorer API",
    description="Backend for the Eatsplorer conversational AI chatbot for Legazpi City dining discovery.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from dotenv import load_dotenv
load_dotenv()

RASA_URL = os.getenv("RASA_URL", "http://localhost:5005")
DB_PATH  = os.path.join(os.path.dirname(__file__), "eatsplorer.db")

# ─────────────────────────────────────────────
# Data Loading — SQLite
# ─────────────────────────────────────────────

def _load_table(table: str) -> pd.DataFrame:
    """Read a table from SQLite and return as DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

# restaurant_scores — pre-computed by infer_restaurants.py
DB: pd.DataFrame = _load_table("restaurant_scores")

# restaurant_information — addresses, cuisine types, etc.
try:
    INFO_DB = _load_table("restaurant_information")
    SUPPORTED_RESTAURANTS = INFO_DB["restaurant_name"].dropna().str.lower().str.strip().tolist()
except Exception as e:
    print(f"Warning: Could not load restaurant_information from DB: {e}")
    INFO_DB = pd.DataFrame()
    SUPPORTED_RESTAURANTS = []

ASPECT_COLS = {
    "food_quality": ("food_quality_avg", "food_quality_polarity", "food_quality_review_count"),
    "service":      ("service_avg",      "service_polarity",      "service_review_count"),
    "ambiance":     ("ambiance_avg",     "ambiance_polarity",     "ambiance_review_count"),
    "price_value":  ("price_value_avg",  "price_value_polarity",  "price_value_review_count"),
    "overall":      ("overall_avg",      "overall_polarity",      "overall_review_count"),
}

# restaurant_scores_monthly — pre-computed per-month scores
try:
    MONTHLY_DB: pd.DataFrame = _load_table("restaurant_scores_monthly")
except Exception as e:
    print(f"Warning: Could not load restaurant_scores_monthly from DB: {e}")
    MONTHLY_DB = pd.DataFrame()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def safe_float(val) -> Optional[float]:
    """Return float or None — handles NaN, None, and strings like 'N/A'."""
    try:
        if val is None:
            return None
        f = float(val)
        if math.isnan(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None

def safe_int(val, default: int = 0) -> int:
    """Return int or default — handles NaN and None."""
    try:
        if val is None:
            return default
        f = float(val)
        if math.isnan(f):
            return default
        return int(f)
    except (TypeError, ValueError):
        return default

def safe_str(val) -> Optional[str]:
    """Return str or None — handles NaN."""
    if val is None:
        return None
    try:
        if isinstance(val, float) and math.isnan(val):
            return None
    except TypeError:
        pass
    s = str(val).strip()
    return None if s in ("", "nan", "None", "N/A") else s

# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    sender: str = "user"
    message: str

class BotResponse(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None
    buttons: Optional[list] = None
    custom: Optional[dict] = None

class AspectScore(BaseModel):
    avg: Optional[float] = None
    polarity: Optional[str] = None
    review_count: int = 0

class Restaurant(BaseModel):
    restaurant_name: str
    food_quality: AspectScore
    service: AspectScore
    ambiance: AspectScore
    price_value: AspectScore
    overall: AspectScore
    composite_score: Optional[float] = None
    composite_polarity: Optional[str] = None
    total_reviews: int = 0
    aspects_scored: int = 0

class StatsResponse(BaseModel):
    total_restaurants: int
    positive_count: int
    neutral_count: int
    negative_count: int
    avg_composite_score: float
    fully_scored_count: int
    aspect_coverage: dict

class MonthlyScore(BaseModel):
    restaurant_name: str
    year: int
    month: int
    food_quality: AspectScore
    service: AspectScore
    ambiance: AspectScore
    price_value: AspectScore
    overall: AspectScore
    composite_score: Optional[float] = None
    composite_polarity: Optional[str] = None
    total_reviews: int = 0
    aspects_scored: int = 0

# ─────────────────────────────────────────────
# Row Serializer
# ─────────────────────────────────────────────

def row_to_restaurant(row) -> Restaurant:
    def asp(key) -> AspectScore:
        avg_col, pol_col, cnt_col = ASPECT_COLS[key]
        return AspectScore(
            avg=safe_float(row.get(avg_col)),
            polarity=safe_str(row.get(pol_col)),
            review_count=safe_int(row.get(cnt_col)),
        )

    return Restaurant(
        restaurant_name=str(row.get("restaurant_name", "")),
        food_quality=asp("food_quality"),
        service=asp("service"),
        ambiance=asp("ambiance"),
        price_value=asp("price_value"),
        overall=asp("overall"),
        composite_score=safe_float(row.get("composite_score")),
        composite_polarity=safe_str(row.get("composite_polarity")),
        total_reviews=safe_int(row.get("total_reviews")),
        aspects_scored=safe_int(row.get("aspects_scored")),
    )

def row_to_monthly(row) -> MonthlyScore:
    def asp(key) -> AspectScore:
        avg_col, pol_col, cnt_col = ASPECT_COLS[key]
        return AspectScore(
            avg=safe_float(row.get(avg_col)),
            polarity=safe_str(row.get(pol_col)),
            review_count=safe_int(row.get(cnt_col)),
        )

    return MonthlyScore(
        restaurant_name=str(row.get("restaurant_name", "")),
        year=safe_int(row.get("Year")),
        month=safe_int(row.get("Month")),
        food_quality=asp("food_quality"),
        service=asp("service"),
        ambiance=asp("ambiance"),
        price_value=asp("price_value"),
        overall=asp("overall"),
        composite_score=safe_float(row.get("composite_score")),
        composite_polarity=safe_str(row.get("composite_polarity")),
        total_reviews=safe_int(row.get("total_reviews")),
        aspects_scored=safe_int(row.get("aspects_scored")),
    )

# ─────────────────────────────────────────────
# Routes — Chat (RASA Proxy)
# ─────────────────────────────────────────────

@app.post("/api/chat", response_model=List[BotResponse])
async def chat(msg: ChatMessage):
    payload = {"sender": msg.sender, "message": msg.message}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{RASA_URL}/webhooks/rest/webhook",
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return [BotResponse(text="I'm sorry, I didn't get a response. Please try again.")]
            return [BotResponse(
                text=item.get("text") or None,
                image=item.get("image"),
                buttons=item.get("buttons"),
                custom=item.get("custom"),
            ) for item in data if item.get("text") or item.get("image") or item.get("custom")]
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="RASA server is not running. Please start it with: rasa run --enable-api --cors '*'"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────
# Routes — Restaurants
# ─────────────────────────────────────────────

@app.get("/api/restaurants", response_model=List[Restaurant])
def get_restaurants(
    aspect: Optional[str] = Query(None),
    polarity: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
    search: Optional[str] = Query(None),
    cuisine_type: Optional[str] = Query(None),
):
    df = DB.copy()

    if cuisine_type and not INFO_DB.empty:
        matching = INFO_DB[
            INFO_DB["cuisine_type"].str.lower().str.contains(cuisine_type.lower(), na=False)
        ]["restaurant_name"].tolist()
        df = df[df["restaurant_name"].isin(matching)]

    if search:
        df = df[df["restaurant_name"].str.lower().str.contains(search.lower(), na=False)]

    if polarity:
        if aspect and aspect in ASPECT_COLS:
            pol_col = ASPECT_COLS[aspect][1]
            df = df[df[pol_col] == polarity]
        else:
            df = df[df["composite_polarity"] == polarity]

    if aspect and aspect in ASPECT_COLS:
        avg_col = ASPECT_COLS[aspect][0]
        df = df[df[avg_col].notna()].sort_values(avg_col, ascending=False)
    else:
        df = df[df["composite_score"].notna()].sort_values("composite_score", ascending=False)

    df = df.head(limit)
    return [row_to_restaurant(row) for _, row in df.iterrows()]


# ─────────────────────────────────────────────
# Routes — Best by Month
# ─────────────────────────────────────────────

@app.get("/api/restaurants/best-by-month", response_model=List[MonthlyScore])
def get_best_by_month(
    year: int = Query(..., description="4-digit year, e.g. 2025"),
    month: int = Query(..., ge=1, le=12, description="Month number 1-12"),
    limit: int = Query(10, ge=1, le=50),
    cuisine_type: Optional[str] = Query(None),
    aspect: Optional[str] = Query(None, description="Sort by this aspect avg instead of composite_score"),
):
    """
    Return the top restaurants for a given year/month, sorted by composite_score
    (or by a specific aspect avg if provided). Optionally filter by cuisine_type.
    """
    if MONTHLY_DB.empty:
        raise HTTPException(status_code=503, detail="Monthly scores database not loaded.")

    df = MONTHLY_DB[(MONTHLY_DB["Year"] == year) & (MONTHLY_DB["Month"] == month)].copy()

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {year}-{month:02d}.")

    # Optional cuisine filter via INFO_DB
    if cuisine_type and not INFO_DB.empty:
        matching = INFO_DB[
            INFO_DB["cuisine_type"].str.lower().str.contains(cuisine_type.lower(), na=False)
        ]["restaurant_name"].tolist()
        df = df[df["restaurant_name"].isin(matching)]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No '{cuisine_type}' restaurants found for {year}-{month:02d}.")

    # Sort
    if aspect and aspect in ASPECT_COLS:
        sort_col = ASPECT_COLS[aspect][0]
        df = df[df[sort_col].notna()].sort_values(sort_col, ascending=False)
    else:
        df = df[df["composite_score"].notna()].sort_values("composite_score", ascending=False)

    df = df.head(limit)
    return [row_to_monthly(row) for _, row in df.iterrows()]


@app.get("/api/restaurants/{name}", response_model=Restaurant)
def get_restaurant(name: str):
    name_lower = name.lower().strip()
    match = DB[DB["restaurant_name"].str.lower() == name_lower]
    if match.empty:
        match = DB[DB["restaurant_name"].str.lower().str.contains(name_lower, na=False)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Restaurant '{name}' not found.")
    return row_to_restaurant(match.iloc[0])


@app.get("/api/stats", response_model=StatsResponse)
def get_stats():
    pol_counts = DB["composite_polarity"].value_counts().to_dict()
    valid_scores = DB["composite_score"].dropna()
    coverage = {}
    for aspect, (avg_col, _, cnt_col) in ASPECT_COLS.items():
        coverage[aspect] = int((DB[cnt_col].fillna(0) > 0).sum())

    return StatsResponse(
        total_restaurants=len(DB),
        positive_count=int(pol_counts.get("Positive", 0)),
        neutral_count=int(pol_counts.get("Neutral", 0)),
        negative_count=int(pol_counts.get("Negative", 0)),
        avg_composite_score=round(float(valid_scores.mean()), 3) if len(valid_scores) > 0 else 0.0,
        fully_scored_count=int((DB["aspects_scored"] == 5).sum()),
        aspect_coverage=coverage,
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "restaurants_loaded": len(DB)}






# ─────────────────────────────────────────────
# Routes — Live Review Lookup (SerpApi)
# ─────────────────────────────────────────────

from live_reviews import get_latest_serpapi_review
from hf_absa import infer_absa

class LiveReviewRequest(BaseModel):
    restaurant_name: str

class SingleReview(BaseModel):
    author: str
    rating: Optional[float] = None
    text: str
    time: str
    absa_inference: Optional[dict] = None # Analysis moves inside here

class LiveReviewResponse(BaseModel):
    found_name: str
    google_rating: Optional[float] = None
    reviews: List[SingleReview] = []  # Changed from 'review' to a list
    summary_text: str
    error: Optional[str] = None

@app.post("/api/live-reviews", response_model=LiveReviewResponse)
async def live_reviews(req: LiveReviewRequest):
    query_name = req.restaurant_name.lower().strip()
    
    # 1. Check supported list
    is_supported = any(query_name in r or r in query_name for r in SUPPORTED_RESTAURANTS)
    if not is_supported:
         return LiveReviewResponse(
            found_name=req.restaurant_name,
            summary_text=f"Sorry, I currently only cover specific partnered or popular restaurants in Legazpi City. '{req.restaurant_name}' is not supported yet!"
        )

    # 2. Fetch 5 reviews (added max_reviews=5)
    result = await get_latest_serpapi_review(req.restaurant_name, max_reviews=5)

    if result["status"] == "no_api_key":
        raise HTTPException(status_code=503, detail="SerpApi key not configured.")
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Could not find '{req.restaurant_name}' on Google Maps.")

    found_name    = result.get("found_name", req.restaurant_name)
    google_rating = result.get("google_rating")
    raw_reviews   = result.get("reviews", []) # Get the list of reviews

    # 3. Handle errors or lack of reviews
    if result["status"] == "error":
        return LiveReviewResponse(
            found_name=found_name,
            summary_text=f"⚠️ Scraper Error: {result.get('message')}",
            reviews=[]
        )

    if not raw_reviews:
        return LiveReviewResponse(
            found_name=found_name,
            summary_text="No recent text reviews were found for this restaurant.",
            reviews=[]
        )

    processed_reviews = []

    # 4. LOOP through the 5 reviews and analyze each one
    for r_data in raw_reviews:
        text = r_data.get("text", "")
        
        # Clean the rating for the AI model
        try:
            raw_val = r_data.get("rating")
            rating_int = int(float(raw_val)) if raw_val is not None else 3
        except:
            rating_int = 3

        # Run ABSA inference for THIS specific review
        absa_result = None
        if text:
            try:
                absa_result = await infer_absa(review_text=text, rating=rating_int)
            except Exception as e:
                print(f"🚨 ABSA Error for {found_name}: {e}")

        # Add to our list
        processed_reviews.append(SingleReview(
            author=r_data["author"],
            rating=r_data.get("rating"),
            text=text,
            time=r_data["time"],
            absa_inference=absa_result
        ))

    # 5. Return the full list
    return LiveReviewResponse(
        found_name=found_name,
        google_rating=google_rating,
        reviews=processed_reviews,
        summary_text=f"Here are the latest 5 reviews for **{found_name}**:"
    )


# ─────────────────────────────────────────────
# Routes — Restaurant Info (address + map embed)
# Missing from main.py — this is what RestaurantInfoPanel.jsx calls
# ─────────────────────────────────────────────

class RestaurantInfo(BaseModel):
    restaurant_name: str
    cuisine_type: Optional[str] = None
    address: Optional[str] = None
    google_maps_url: Optional[str] = None
    google_maps_embed_url: Optional[str] = None
    best_dish_signature_dish: Optional[str] = None
    quick_summary: Optional[str] = None

def _make_embed_url(address: str, name: str) -> str:
    import urllib.parse
    # Include restaurant name + address so Google Maps pins the exact business.
    # Base format maps.google.com/maps?q=...&output=embed is the only one
    # browsers allow in iframes without an API key.
    if address:
        q = f"{name}, {address}"
    else:
        q = f"{name}, Legazpi City, Albay, Philippines"
    return f"https://maps.google.com/maps?q={urllib.parse.quote(q)}&output=embed"

def _row_to_info(row) -> RestaurantInfo:
    name     = str(row.get("restaurant_name", ""))
    # DB column is 'google_maps_address' — covers both address and maps URL
    address  = safe_str(row.get("google_maps_address") or row.get("address"))
    maps_url = safe_str(row.get("google_maps_url") or row.get("google_maps_address"))
    return RestaurantInfo(
        restaurant_name=name,
        cuisine_type=safe_str(row.get("cuisine_type")),
        address=address,
        google_maps_url=maps_url,
        google_maps_embed_url=_make_embed_url(address or "", name),
        best_dish_signature_dish=safe_str(row.get("best_dish_signature_dish")),
        quick_summary=safe_str(row.get("quick_summary")),
    )

@app.get("/api/restaurant-info/{name}", response_model=RestaurantInfo)
def get_restaurant_info(name: str):
    try:
        if INFO_DB.empty:
            raise HTTPException(status_code=503, detail="Restaurant info database not loaded.")
        q = name.lower().strip()
        match = INFO_DB[INFO_DB["restaurant_name"].str.lower().str.strip() == q]
        if match.empty:
            match = INFO_DB[INFO_DB["restaurant_name"].str.lower().str.contains(q, na=False, regex=False)]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"No info found for '{name}'.")
        return _row_to_info(match.iloc[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/restaurant-info", response_model=List[RestaurantInfo])
def get_all_restaurant_info():
    try:
        if INFO_DB.empty:
            return []
        return [_row_to_info(row) for _, row in INFO_DB.iterrows()]
    except Exception:
        return []


# ─────────────────────────────────────────────
# Routes — Live Reviews Bulk (10 reviews + ABSA each)
# ─────────────────────────────────────────────

class BulkReviewItem(BaseModel):
    author: str
    rating: Optional[int] = None
    text: str
    time: str
    absa_inference: Optional[dict] = None

class LiveReviewsBulkResponse(BaseModel):
    found_name: str
    google_rating: Optional[float] = None
    reviews: List[BulkReviewItem] = []
    error: Optional[str] = None

@app.post("/api/live-reviews-bulk", response_model=LiveReviewsBulkResponse)
async def live_reviews_bulk(req: LiveReviewRequest):
    """
    Fetches up to 10 recent Google Maps reviews via SerpApi and runs
    ABSA inference on each one. Used by the Live Reviews panel in cards.
    """
    query_name = req.restaurant_name.lower().strip()
    is_supported = any(query_name in r or r in query_name for r in SUPPORTED_RESTAURANTS)

    if not is_supported:
        return LiveReviewsBulkResponse(
            found_name=req.restaurant_name,
            error=f"'{req.restaurant_name}' is not in our supported restaurant list yet."
        )

    result = await get_latest_serpapi_review(req.restaurant_name, max_reviews=10)

    if result["status"] in ("error", "not_found", "no_reviews"):
        return LiveReviewsBulkResponse(
            found_name=result.get("found_name", req.restaurant_name),
            error=result.get("message") or f"No reviews found for '{req.restaurant_name}'."
        )

    found_name    = result.get("found_name", req.restaurant_name)
    google_rating = result.get("google_rating")
    raw_reviews   = result.get("reviews", [])

    bulk_items = []
    for rv in raw_reviews:
        absa_result = None
        if rv.get("text"):
            try:
                raw_rating = rv.get("rating")
                try:
                    rating_int = int(float(raw_rating)) if raw_rating is not None else 3
                except (ValueError, TypeError):
                    rating_int = 3
                absa_result = await infer_absa(review_text=rv["text"], rating=rating_int)
            except Exception as e:
                print(f"ABSA error: {e}")
        bulk_items.append(BulkReviewItem(
            author=rv.get("author", "Google User"),
            rating=rv.get("rating"),
            text=rv.get("text", ""),
            time=rv.get("time", "Recently"),
            absa_inference=absa_result,
        ))

    return LiveReviewsBulkResponse(
        found_name=found_name,
        google_rating=google_rating,
        reviews=bulk_items,
    )