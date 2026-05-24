"""
Eatsplorer HuggingFace ABSA Inference Module
Uses Gradio API from HuggingFace Spaces instead of loading the model locally.
No torch or transformers required on the local machine.
"""

import os
from gradio_client import Client

# note: preprocessing (demojize + rating prefix) is handled by the HF space
# no local torch or transformers needed

# -- Config ------------------------------------------------------------------

SPACE_ID  = "Your HF Space ID"   # your HF Space name
HF_TOKEN  = os.getenv("HF_TOKEN")        # optional, only needed if space is private

ASPECTS   = ["overall", "food quality", "service", "ambiance", "price value"]
ID2LABEL  = {0: "N/A", 1: "Negative", 2: "Neutral", 3: "Positive"}

# -- Client init -------------------------------------------------------------

print("connecting to eatsplorer gradio space...")

try:
    client = Client(SPACE_ID, HF_TOKEN)
    print(f"connected to space: {SPACE_ID}")
except Exception as e:
    print(f"failed to connect to gradio space: {e}")
    client = None

# -- Inference ---------------------------------------------------------------

async def infer_absa(review_text: str, rating: int = 3):
    """
    analyzes sentiment across 5 aspects via gradio api (overall, food quality, service, ambiance, price value).
    rating: star rating 1-5 from the review source (defaults to 3 if unknown).
    """
    if client is None:
        print("cannot run inference: gradio client not connected.")
        return None

    try:
        # calls the /infer_absa endpoint on your HF space
        # pass raw text + rating separately; the space handles preprocessing
        result = client.predict(
            review_text=review_text,
            rating=float(rating),
            api_name="/infer_absa"
        )

        # result is expected to be a dict like:
        # {
        #   "overall":      {"sentiment": "Positive", "score": 4.5},
        #   "food_quality": {"sentiment": "Positive", "score": 4.2},
        #   "service":      {"sentiment": "Neutral",  "score": 3.0},
        #   "ambiance":     {"sentiment": "Positive", "score": 4.5},
        #   "price_value":  {"sentiment": "Negative", "score": 1.8},
        # }

        if not isinstance(result, dict):
            print(f"unexpected response format from space: {result}")
            return None

        return result

    except Exception as e:
        print(f"gradio inference error: {e}")
        return None