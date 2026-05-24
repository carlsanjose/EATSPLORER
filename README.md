# Eatsplorer: Enhancing Food Tourism in Legazpi City Through a Conversational AI Chatbot Using Aspect-Based Sentiment Analysis

**Bicol University College of Science — Computer Science Department**
**Bachelor of Science in Computer Science | Undergraduate Thesis | May 2026**

**Authors:** Manuel James L. Baltasar · Lance Andrei C. Bolaños · Carl Edward L. San Jose

---

## Overview

Eatsplorer is a conversational AI chatbot that addresses fragmented dining discovery in Legazpi City, Philippines by delivering aspect-based restaurant recommendations grounded in real customer sentiment — going beyond generic star ratings. Users can query the chatbot in natural language to get restaurant recommendations filtered by cuisine type, specific dining aspects (food quality, service, ambiance, price/value), or signature dishes, and can retrieve live Google Maps reviews analyzed in real time.

The system integrates two core AI modules:

- **Fine-tuned DeBERTa V3 ABSA Model** — a pre-trained `deberta-v3-base-absa-v1.1` model fine-tuned on over 10,000 curated Legazpi City restaurant reviews using a class-weighted loss function and mixed-precision training. It classifies sentiment across five aspects (Food Quality, Service, Ambiance, Price/Value, Overall) with labels Positive, Neutral, Negative, and N/A. Achieved a peak Macro F1-score of **0.9699**.

- **RASA-Based Conversational Pipeline** — a RASA 3.x NLU pipeline using DIETClassifier with SpaCy featurizers, trained for intent classification (Macro F1: **0.8464**) and entity extraction (Macro F1: **0.8950**) over restaurant-domain conversational queries including temporal filtering.

The system achieved a **SUS score of 79.60/100** and an **87.1% overall satisfaction rate** during User Acceptance Testing.

📄 **Final Manuscript (PDF):** [View on Google Drive](https://drive.google.com/file/d/1ezr_2qf6hIpCt64aGN40nnOp36IX-qsW/view?usp=drive_link)

🌐 **Live Frontend Demo:** [eatsplorerchatbotfrontend.vercel.app](https://eatsplorerchatbotfrontend.vercel.app/)

🤗 **ABSA Model (Hugging Face):** [Ehemjayy/EATSPLORER_REVISED](https://huggingface.co/Ehemjayy/EATSPLORER_REVISED/tree/main)

🚀 **ABSA Inference Space (Hugging Face Spaces):** [Ehemjayy/EATSPLORER](https://huggingface.co/spaces/Ehemjayy/EATSPLORER)

---

## System Architecture

```
React Frontend (Vercel / Localhost:5173)
        │
        ▼ HTTP
FastAPI Backend (Localhost:8000)  ──── SerpAPI (Live Google Maps Reviews)
        │                                      │
        ▼ HTTP                                 ▼ Gradio API
RASA Server (Localhost:5005)        HuggingFace Spaces (DeBERTa V3 ABSA)
        │
        ▼ HTTP
RASA Actions Server (Localhost:5055)
        │
        ▼ SQL
SQLite Database
```

> For public access during local development, the FastAPI and RASA servers are exposed via **Ngrok** tunnels, and the React frontend is pointed to those tunnel URLs.

---

## Dependencies and Versions

Two separate Python virtual environments are required due to a `pydantic` v1/v2 conflict between RASA and FastAPI.

### Python

| Requirement | Version |
|---|---|
| Python (both venvs) | `3.10.x` |

### `venv_chatbot` — RASA + Actions Server

| Package | Version |
|---|---|
| `rasa` | `3.6.20` |
| `spacy` | `3.8.7` |
| `en_core_web_lg` (spaCy model) | via `python -m spacy download en_core_web_lg` |

### `venv_web` — FastAPI Backend + ABSA Utilities

| Package | Version |
|---|---|
| `fastapi` | `0.121.0` |
| `uvicorn` | latest stable |
| `httpx` | latest stable |
| `transformers` | `5.1.0` |
| `torch` | `2.7.1+cu118` (GPU) or `2.7.1` (CPU) |
| `pandas` | `2.0.3` |
| `scikit-learn` | `1.1.3` |
| `emoji` | `2.15.0` |
| `gradio_client` | latest stable |
| `python-dotenv` | latest stable |

### Frontend

| Requirement | Version |
|---|---|
| Node.js | `18.x` or later |
| React | `18.3.1` |
| Vite | `5.3.4` |

---

## Local Installation and Deployment

### Prerequisites

- Python 3.10 installed and available as `python3.10`
- Node.js 18+ and npm installed
- [Ngrok](https://ngrok.com/) installed and authenticated (for public tunneling)
- A [SerpAPI](https://serpapi.com/) API key (for the live reviews feature)

---

### 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/eatsplorer.git
cd eatsplorer
```

---

### 2. Set Up Environment Variables

Create a `.env` file in the backend root directory:

```env
SERPAPI_API_KEY=your_serpapi_key_here
RASA_URL=http://localhost:5005
HF_SPACE_URL=https://ehemjayy-eatsplorer.hf.space
```

---

### 3. Set Up the RASA Virtual Environment (`venv_chatbot`)

```bash
# Create and activate the RASA venv
python3.10 -m venv venv_chatbot
source venv_chatbot/bin/activate        # Windows: venv_chatbot\Scripts\activate

# Install RASA and SpaCy
pip install rasa==3.6.20
pip install spacy==3.8.7
python -m spacy download en_core_web_lg

deactivate
```

---

### 4. Set Up the FastAPI Virtual Environment (`venv_web`)

```bash
# Create and activate the web/backend venv
python3.10 -m venv venv_web
source venv_web/bin/activate            # Windows: venv_web\Scripts\activate

# Install backend dependencies
pip install fastapi==0.121.0 uvicorn httpx python-dotenv
pip install transformers==5.1.0
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
# For CPU-only machines:
# pip install torch==2.7.1
pip install pandas==2.0.3 scikit-learn==1.1.3 emoji==2.15.0 gradio_client

deactivate
```

---

### 5. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

### 6. Running the System Locally

All four components must be running simultaneously. Open four separate terminal windows.

**Terminal 1 — RASA NLU/Dialogue Server**

```bash
cd chatbot
source ../venv_chatbot/bin/activate     # Windows: ..\venv_chatbot\Scripts\activate
rasa run --enable-api --cors "*" --port 5005
```

**Terminal 2 — RASA Custom Actions Server**

```bash
cd chatbot
source ../venv_chatbot/bin/activate
rasa run actions --port 5055
```

**Terminal 3 — FastAPI Backend**

```bash
cd backend
source ../venv_web/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 4 — React Frontend (Development Server)**

```bash
cd frontend
npm run dev
```

The frontend will be accessible at `http://localhost:5173`.

---

### 7. Exposing the Backend via Ngrok (Optional — for Remote Access)

If you need to access the chatbot from outside your local machine (e.g., from the deployed Vercel frontend), expose both the FastAPI and RASA servers via Ngrok tunnels:

```bash
# Expose FastAPI backend
ngrok http 8000

# In a separate terminal, expose the RASA server
ngrok http 5005
```

Update the `.env` file (and Vercel environment variables for the deployed frontend) with the generated Ngrok URLs.

---

### 8. Retraining the RASA Pipeline

After modifying training data in `chatbot/data/` (NLU, stories, rules) or the domain in `chatbot/domain.yml`:

```bash
cd chatbot
source ../venv_chatbot/bin/activate
rasa train
```

The newly trained model will be saved to `chatbot/models/`. Restart the RASA server to load it.

---

## Fine-Tuned DeBERTa V3 ABSA Model

### Model Details

- **Base Model:** `yangheng/deberta-v3-base-absa-v1.1`
- **Task:** Aspect Category Sentiment Analysis (ACSA)
- **Aspects:** Food Quality, Service, Ambiance, Price/Value, Overall
- **Labels:** Positive, Neutral, Negative, N/A
- **Training Configuration:** Class-weighted cross-entropy loss, mixed-precision (fp16), 5-fold cross-validation
- **Peak Macro F1:** 0.9699

### Hugging Face Links

- **Model Repository:** [Ehemjayy/EATSPLORER_REVISED](https://huggingface.co/Ehemjayy/EATSPLORER_REVISED/tree/main)
- **Inference Space (Gradio API):** [Ehemjayy/EATSPLORER](https://huggingface.co/spaces/Ehemjayy/EATSPLORER)

---

### Using the Hosted Inference Space

The easiest way to run ABSA inference is through the Gradio API hosted on Hugging Face Spaces. No local GPU is required.

```python
from gradio_client import Client

client = Client("Ehemjayy/EATSPLORER")

result = client.predict(
    review_text="The food was amazing but the service was a bit slow.",
    rating=4.0,
    api_name="/infer_absa"
)

# result is a dict of aspect -> {"sentiment": str, "score": float}
print(result)
```

---

### Running Inference Locally

To run inference locally using the downloaded model weights:

```bash
source venv_web/bin/activate
python infer_restaurants.py --input data/reviews.csv --output data/scored_reviews.csv
```

Or for a single review:

```bash
python score_reviews.py --text "Great sinigang, very affordable." --rating 5
```

---

### Retraining / Fine-Tuning the Model

To fine-tune the DeBERTa V3 model on new or updated review data:

1. Prepare your dataset in the ACSA format (see `data/eatsplorer_master.csv` for reference structure — columns: `text`, `rating`, `food_quality`, `service`, `ambiance`, `price_value`, `overall`).

2. Activate the web venv and run the training script:

```bash
source venv_web/bin/activate
python train_absa.py \
  --data_path data/eatsplorer_master.csv \
  --model_output models/deberta_absa_finetuned \
  --epochs 5 \
  --batch_size 16 \
  --fp16
```

3. GPU with at least 4 GB VRAM is recommended (NVIDIA GTX 1660 Ti or better). CPU training is supported but significantly slower.

4. After training, update the model on Hugging Face using the `huggingface_hub` CLI:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload Ehemjayy/EATSPLORER_REVISED ./models/deberta_absa_finetuned
```

---

## Citation

> **Note:** This section will be finalized once the final edited manuscript has been officially uploaded. Please use the placeholder below in the meantime.

If you use Eatsplorer, its dataset, or the fine-tuned ABSA model in academic work, please cite:

**APA:**

> Baltasar, M. J. L., Bolaños, L. A. C., & San Jose, C. E. L. (2026). *Eatsplorer: Enhancing food tourism in Legazpi City through a conversational AI chatbot using Aspect-Based Sentiment Analysis* [Unpublished undergraduate thesis]. Bicol University College of Science.

**BibTeX:**

```bibtex
@thesis{eatsplorer2026,
  author    = {Baltasar, Manuel James L. and Bola{\~n}os, Lance Andrei C. and {San Jose}, Carl Edward L.},
  title     = {Eatsplorer: Enhancing Food Tourism in {Legazpi City} Through a Conversational {AI} Chatbot Using Aspect-Based Sentiment Analysis},
  school    = {Bicol University College of Science},
  year      = {2026},
  type      = {Unpublished Undergraduate Thesis},
  note      = {Manuscript available at: \url{https://drive.google.com/file/d/1ezr_2qf6hIpCt64aGN40nnOp36IX-qsW/view?usp=drive_link}}
}
```

---

## License

This project is released for academic and non-commercial use. For other uses, please contact the authors.
