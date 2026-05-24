import os
import httpx

SERPAPI_API_KEY = "PASTE YOUR SERPAPI KEY HERE"  # or set as environment variable SERPAPI_API_KEY

"""
Live review fetcher for Eatsplorer. ABSA aspects expected: overall, food_quality, service, ambiance, price_value.
"""
async def get_latest_serpapi_review(restaurant_name: str, max_reviews: int = 10) -> dict:
    if not SERPAPI_API_KEY or SERPAPI_API_KEY == "paste_your_serpapi_key_here":
        return {"status": "error", "message": "SerpApi Key is missing."}

    query = f"{restaurant_name} Legazpi City Philippines"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:

            # STEP 1: Find the place's data_id
            search_params = {
                "engine": "google_maps",
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "hl": "en",
            }
            search_resp = await client.get("https://serpapi.com/search.json", params=search_params)
            search_resp.raise_for_status()
            search_data = search_resp.json()

            if "error" in search_data:
                return {"status": "error", "message": f"SerpApi Error: {search_data['error']}"}

            place_results = search_data.get("place_results", {})
            if not place_results or "data_id" not in place_results:
                local = search_data.get("local_results", [])
                if not local or not isinstance(local, list):
                    return {"status": "not_found", "found_name": restaurant_name}
                place_results = local[0]

            if not isinstance(place_results, dict) or "data_id" not in place_results:
                return {"status": "not_found", "found_name": restaurant_name}

            data_id      = place_results["data_id"]
            found_name   = place_results.get("title", restaurant_name)
            google_rating = place_results.get("rating")

            # STEP 2: Fetch up to max_reviews recent reviews
            reviews_params = {
                "engine":   "google_maps_reviews",
                "data_id":  data_id,
                "api_key":  SERPAPI_API_KEY,
                "hl":       "en",
                "sort_by":  "newestFirst",
            }
            reviews_resp = await client.get("https://serpapi.com/search.json", params=reviews_params)
            reviews_resp.raise_for_status()
            reviews_data = reviews_resp.json()

            all_reviews = reviews_data.get("reviews", [])
            if not isinstance(all_reviews, list):
                all_reviews = []

            text_reviews = [
                r for r in all_reviews
                if isinstance(r, dict) and r.get("snippet")
            ]

            if not text_reviews:
                return {"status": "no_reviews", "found_name": found_name, "google_rating": google_rating}

            # Return up to max_reviews, each as a review dict
            reviews_out = []
            for r in text_reviews[:max_reviews]:
                user_info = r.get("user", {})
                if not isinstance(user_info, dict):
                    user_info = {}
                reviews_out.append({
                    "author": user_info.get("name", "Google User"),
                    "rating": r.get("rating"),
                    "text":   r.get("snippet", ""),
                    "time":   r.get("date", "Recently"),
                })

            return {
                "status":        "ok",
                "found_name":    found_name,
                "google_rating": google_rating,
                # Keep single "review" for backwards compat with existing chat action
                "review":        reviews_out[0] if reviews_out else None,
                # New: full list for the live reviews panel
                "reviews":       reviews_out,
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}
