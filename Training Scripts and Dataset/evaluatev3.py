import os
import numpy as np
import pandas as pd
import torch
import emoji
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_DIR  = "./eatsplorer_finetuned_model"
CSV_FILE   = "training_dataset_revision.csv"
OUTPUT_DIR = "./evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL2ID = {"N/A": 0, "Negative": 1, "Neutral": 2, "Positive": 3}
ID2LABEL  = {0: "N/A", 1: "Negative", 2: "Neutral", 3: "Positive"}
ASPECTS   = ["food quality", "service", "ambiance", "price value", "overall"]

THRESHOLDS = {
    "accuracy":    0.70,
    "f1_macro":    0.60,
    "f1_negative": 0.50,
    "f1_neutral":  0.50,
}

# ==========================================
# SANITY REVIEWS
# ==========================================
SANITY_REVIEWS = [

    # ── 1 ASPECT ──────────────────────────────────────────────────────────────
    (
        "Bicol Express is their best dish! Authentic and creamy.",
        5,
        {"food quality": "Positive", "service": "N/A", "ambiance": "N/A", "price value": "N/A", "overall": "Positive"}
    ),
    (
        "Terrible food. The Laing was watery and had no coconut milk flavor.",
        1,
        {"food quality": "Negative", "service": "N/A", "ambiance": "N/A", "price value": "N/A", "overall": "Negative"}
    ),
    (
        "The staff were incredibly friendly and attentive throughout our visit.",
        5,
        {"food quality": "N/A", "service": "Positive", "ambiance": "N/A", "price value": "N/A", "overall": "Positive"}
    ),
    (
        "Waited 45 minutes and nobody came to take our order. Bastos ang staff.",
        1,
        {"food quality": "N/A", "service": "Negative", "ambiance": "N/A", "price value": "N/A", "overall": "Negative"}
    ),
    (
        "Beautiful place! The view of Mayon Volcano is breathtaking.",
        5,
        {"food quality": "N/A", "service": "N/A", "ambiance": "Positive", "price value": "N/A", "overall": "Positive"}
    ),
    (
        "The restaurant was filthy. Tables were sticky and floors unswept.",
        1,
        {"food quality": "N/A", "service": "N/A", "ambiance": "Negative", "price value": "N/A", "overall": "Negative"}
    ),
    (
        "Best value for money in all of Legazpi.",
        5,
        {"food quality": "N/A", "service": "N/A", "ambiance": "N/A", "price value": "Positive", "overall": "Positive"}
    ),
    (
        "Overpriced for what you get. Not worth what you paid.",
        1,
        {"food quality": "N/A", "service": "N/A", "ambiance": "N/A", "price value": "Negative", "overall": "Negative"}
    ),

    # ── NEUTRAL (1 ASPECT) ─────────────────────────────────────────────────────
    (
        "It was Okay. Nothing special but nothing bad either.",
        3,
        {"food quality": "Neutral", "service": "N/A", "ambiance": "N/A", "price value": "N/A", "overall": "Neutral"}
    ),
    (
        "Service was fine. Nothing to rave about, nothing to complain about.",
        3,
        {"food quality": "N/A", "service": "Neutral", "ambiance": "N/A", "price value": "N/A", "overall": "Neutral"}
    ),
    (
        "Normal atmosphere. Standard Filipino restaurant, nothing special.",
        3,
        {"food quality": "N/A", "service": "N/A", "ambiance": "Neutral", "price value": "N/A", "overall": "Neutral"}
    ),
    (
        "Average pricing for Legazpi. Okay naman ang presyo.",
        3,
        {"food quality": "N/A", "service": "N/A", "ambiance": "N/A", "price value": "Neutral", "overall": "Neutral"}
    ),

    # ── 2 ASPECTS ─────────────────────────────────────────────────────────────
    (
        "Food was tasty but so expensive. Not worth it.",
        3,
        {"food quality": "Positive", "service": "N/A", "ambiance": "N/A", "price value": "Negative", "overall": "Neutral"}
    ),
    (
        "The food was bland but the staff were very friendly and accommodating.",
        2,
        {"food quality": "Negative", "service": "Positive", "ambiance": "N/A", "price value": "N/A", "overall": "Negative"}
    ),
    (
        "Great service and very affordable prices. Will definitely come back!",
        5,
        {"food quality": "N/A", "service": "Positive", "ambiance": "N/A", "price value": "Positive", "overall": "Positive"}
    ),
    (
        "Beautiful restaurant but way too expensive for the quality.",
        3,
        {"food quality": "N/A", "service": "N/A", "ambiance": "Positive", "price value": "Negative", "overall": "Neutral"}
    ),
    (
        "The Bicol Express was amazing but the place was dirty and noisy.",
        3,
        {"food quality": "Positive", "service": "N/A", "ambiance": "Negative", "price value": "N/A", "overall": "Neutral"}
    ),
    (
        "Rude staff but at least the prices were very affordable.",
        2,
        {"food quality": "N/A", "service": "Negative", "ambiance": "N/A", "price value": "Positive", "overall": "Negative"}
    ),

    # ── 3 ASPECTS ─────────────────────────────────────────────────────────────
    (
        "Great food and friendly staff but the place was too noisy and cramped.",
        3,
        {"food quality": "Positive", "service": "Positive", "ambiance": "Negative", "price value": "N/A", "overall": "Neutral"}
    ),
    (
        "Their Pinangat was amazing, fast service, and the price was ok.",
        3,
        {"food quality": "Positive", "service": "Positive", "ambiance": "N/A", "price value": "Neutral", "overall": "Neutral"}
    ),
    (
        "Terrible food and rude staff but the beautiful Mayon view saved it.",
        2,
        {"food quality": "Negative", "service": "Negative", "ambiance": "Positive", "price value": "N/A", "overall": "Negative"}
    ),
    (
        "The place is beautiful and very affordable but the food was disappointing.",
        3,
        {"food quality": "Negative", "service": "N/A", "ambiance": "Positive", "price value": "Positive", "overall": "Neutral"}
    ),

    # ── ALL 4 ASPECTS ──────────────────────────────────────────────────────────
    (
        "Yummy food, friendly staff, beautiful place, and great value!",
        5,
        {"food quality": "Positive", "service": "Positive", "ambiance": "Positive", "price value": "Positive", "overall": "Positive"}
    ),
    (
        "Terrible food, rude staff, dirty place, and outrageously overpriced.",
        1,
        {"food quality": "Negative", "service": "Negative", "ambiance": "Negative", "price value": "Negative", "overall": "Negative"}
    ),
    (
        "The food was okay, service was okay, place was okay, prices were okay.",
        3,
        {"food quality": "Neutral", "service": "Neutral", "ambiance": "Neutral", "price value": "Neutral", "overall": "Neutral"}
    ),
    (
        "Great food and beautiful ambiance but rude staff and overpriced.",
        3,
        {"food quality": "Positive", "service": "Negative", "ambiance": "Positive", "price value": "Negative", "overall": "Neutral"}
    ),
    (
        "Bland food and terrible service but the clean place and fair price helped.",
        2,
        {"food quality": "Negative", "service": "Negative", "ambiance": "Positive", "price value": "Positive", "overall": "Negative"}
    ),
]

# ==========================================
# HELPERS
# ==========================================
def preprocess(text, rating):
    text = emoji.demojize(str(text))
    return f"[RATING={rating}] {text}"

def predict_batch(model, tokenizer, aspects, texts, device, batch_size=16):
    all_preds = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            text=aspects[i:i+batch_size],
            text_pair=texts[i:i+batch_size],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return all_preds

# ==========================================
# CHECK 1 — MODEL LOADS CORRECTLY
# ==========================================
def check_model_loads():
    print("=" * 65)
    print("CHECK 1: Model sanity check")
    print("=" * 65)

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"\n❌ Model folder '{MODEL_DIR}' not found.\n"
            f"   Training may not have completed."
        )

    for f in ["config.json", "tokenizer_config.json"]:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            raise FileNotFoundError(f"\n❌ Missing '{f}' — model was not saved correctly.")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    enc = tokenizer(
        text=["food quality"],
        text_pair=["[RATING=5] Sobrang sarap ng pagkain!"],
        return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)
    with torch.no_grad():
        out = model(**enc)

    assert out.logits.shape == (1, 4), "Unexpected output shape"
    print(f"   ✅ Model loaded from  : {MODEL_DIR}")
    print(f"   ✅ Forward pass OK    : test output = '{ID2LABEL[out.logits.argmax().item()]}'")
    print(f"   ✅ Running on         : {str(device).upper()}")
    return model, tokenizer, device

# ==========================================
# CHECK 2 — MULTI-ASPECT SANITY TEST
#   Now also saves results to CSV instead
#   of only printing them.
# ==========================================
def check_sanity(model, tokenizer, device):
    print("\n" + "=" * 65)
    print("CHECK 2: Multi-aspect sanity test")
    print("=" * 65)
    print(f"   Testing {len(SANITY_REVIEWS)} reviews × 5 aspects = {len(SANITY_REVIEWS)*5} predictions\n")

    flat_aspects  = []
    flat_texts    = []
    flat_expected = []
    flat_review_i = []

    for i, (text, rating, aspect_map) in enumerate(SANITY_REVIEWS):
        processed = preprocess(text, rating)
        for aspect in ASPECTS:
            flat_aspects.append(aspect)
            flat_texts.append(processed)
            flat_expected.append(aspect_map[aspect])
            flat_review_i.append(i)

    flat_preds = [ID2LABEL[p] for p in
                  predict_batch(model, tokenizer, flat_aspects, flat_texts, device)]

    total_correct  = 0
    total_pairs    = 0
    csv_rows       = []

    for i, (text, rating, aspect_map) in enumerate(SANITY_REVIEWS):
        short   = text[:55] + "..." if len(text) > 55 else text
        indices = [j for j, ri in enumerate(flat_review_i) if ri == i]
        r_correct    = 0
        r_total      = 0
        aspect_lines = []

        for j in indices:
            asp  = flat_aspects[j]
            exp  = flat_expected[j]
            pred = flat_preds[j]
            ok   = pred == exp
            r_correct += ok
            r_total   += 1
            icon = "✅" if ok else "❌"
            aspect_lines.append(f"      {asp:<14} expected={exp:<10} got={pred:<10} {icon}")

            # Collect row for CSV
            csv_rows.append({
                "review_text":      text,
                "rating":           rating,
                "aspect":           asp,
                "expected":         exp,
                "predicted":        pred,
                "correct":          ok,
            })

        total_correct += r_correct
        total_pairs   += r_total

        status = "✅ ALL CORRECT" if r_correct == r_total else f"⚠️  {r_correct}/{r_total} correct"
        print(f"   [{rating}★] \"{short}\"")
        print(f"          {status}")
        for line in aspect_lines:
            print(line)
        print()

    pct = total_correct / total_pairs * 100
    print(f"   {'─'*55}")
    print(f"   Overall: {total_correct}/{total_pairs} aspect-sentiment pairs correct ({pct:.0f}%)")

    if pct >= 85:
        print("   ✅ PASS — model handles single and multi-aspect reviews well")
    elif pct >= 65:
        print("   ⚠️  PARTIAL — check which aspects/sentiments are failing above")
    else:
        print("   ❌ FAIL — model is struggling with basic predictions")

    # ── Save sanity results to CSV ─────────────────────────────────────────────
    sanity_csv = pd.DataFrame(csv_rows)
    sanity_path = os.path.join(OUTPUT_DIR, "sanity_results.csv")
    sanity_csv.to_csv(sanity_path, index=False)
    print(f"\n   💾 Sanity results saved → {sanity_path}")

# ==========================================
# CHECK 3 — FULL VALIDATION SET METRICS
# ==========================================
def check_full_metrics(model, tokenizer, device):
    print("\n" + "=" * 65)
    print("CHECK 3: Full validation set metrics")
    print("=" * 65)

    if not os.path.exists(CSV_FILE):
        print(f"   ⚠️  {CSV_FILE} not found — skipping.")
        return None, None, None

    df = pd.read_csv(CSV_FILE, keep_default_na=False)
    # Normalize casing — CSV has Title Case aspects and a few lowercase sentiments
    df['aspect']    = df['aspect'].str.strip().str.lower()
    df['sentiment'] = df['sentiment'].str.strip().str.capitalize()
    df['sentiment'] = df['sentiment'].replace({'': 'N/A', 'N/a': 'N/A'})
    df = df[df['sentiment'].isin(LABEL2ID.keys())]

    labels_all = df['sentiment'].map(LABEL2ID).astype(int).tolist()
    _, val_idx = train_test_split(
        range(len(df)), test_size=0.3, random_state=42, stratify=labels_all
    )
    val_df = df.iloc[list(val_idx)].reset_index(drop=True)
    print(f"   Evaluating on {len(val_df):,} validation samples...")

    aspects = val_df['aspect'].tolist()
    texts   = [preprocess(r['review_text'], r['rating']) for _, r in val_df.iterrows()]
    labels  = val_df['sentiment'].map(LABEL2ID).astype(int).tolist()
    preds   = predict_batch(model, tokenizer, aspects, texts, device, batch_size=32)

    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average='macro')
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec  = recall_score(labels, preds, average='macro', zero_division=0)

    def status(score, thresh):
        return "✅ PASS" if score >= thresh else "❌ FAIL"

    print(f"\n   {'Metric':<22} {'Score':<10} {'Min':<8} Status")
    print(f"   {'─'*50}")
    print(f"   {'Accuracy':<22} {acc:.4f}     {THRESHOLDS['accuracy']:.2f}     {status(acc, THRESHOLDS['accuracy'])}")
    print(f"   {'F1 (macro)':<22} {f1:.4f}     {THRESHOLDS['f1_macro']:.2f}     {status(f1, THRESHOLDS['f1_macro'])}")
    print(f"   {'Precision (macro)':<22} {prec:.4f}")
    print(f"   {'Recall (macro)':<22} {rec:.4f}")

    majority_class = max(set(labels), key=labels.count)
    baseline_f1    = f1_score(labels, [majority_class]*len(labels), average='macro', zero_division=0)
    improvement    = (f1 - baseline_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0
    print(f"\n   Majority class baseline F1 : {baseline_f1:.4f}")
    if f1 > baseline_f1:
        print(f"   ✅ Beats baseline by        : +{improvement:.1f}%")
    else:
        print(f"   ❌ Does NOT beat baseline — something went wrong during training")

    return labels, preds, val_df

# ==========================================
# CHECK 4 — PER CLASS BREAKDOWN
# ==========================================
def check_per_class(labels, preds):
    print("\n" + "=" * 65)
    print("CHECK 4: Per-class breakdown")
    print("=" * 65)

    report = classification_report(
        labels, preds,
        target_names=list(ID2LABEL.values()),
        output_dict=True, zero_division=0
    )

    print(f"\n   {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}")
    print(f"   {'─'*58}")

    all_pass = True
    for label_id, name in ID2LABEL.items():
        r         = report[name]
        f1_thresh = THRESHOLDS.get(f'f1_{name.lower()}', 0.50)
        ok        = "✅" if r['f1-score'] >= f1_thresh else "⚠️ "
        if r['f1-score'] < f1_thresh:
            all_pass = False
        print(f"   {name:<12} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1-score']:<12.4f} {int(r['support'])}  {ok}")

    if all_pass:
        print("\n   ✅ All classes meet minimum F1 threshold")
    else:
        print("\n   ⚠️  Weak classes detected — these sentiments need more data or epochs")

# ==========================================
# CHECK 5 — PER ASPECT BREAKDOWN
# ==========================================
def check_per_aspect(labels, preds, val_df):
    print("\n" + "=" * 65)
    print("CHECK 5: Per-aspect breakdown")
    print("=" * 65)

    val_df         = val_df.copy()
    val_df['label'] = labels
    val_df['pred']  = preds

    print(f"\n   {'Aspect':<16} {'Accuracy':<12} {'F1 (macro)':<12} {'Samples'}")
    print(f"   {'─'*52}")

    for aspect in ASPECTS:
        sub = val_df[val_df['aspect'] == aspect]
        if len(sub) == 0:
            continue
        acc = accuracy_score(sub['label'], sub['pred'])
        f1  = f1_score(sub['label'], sub['pred'], average='macro', zero_division=0)
        ok  = "✅" if f1 >= THRESHOLDS['f1_macro'] else "⚠️ "
        print(f"   {aspect:<16} {acc:<12.4f} {f1:<12.4f} {len(sub)}  {ok}")

# ==========================================
# CHECK 6 — CHARTS
#   Now includes:
#     • Overall confusion matrix
#     • Per-aspect confusion matrices (2×2 grid)
#     • Per-class F1 bar chart
# ==========================================
def check_confusion_and_charts(labels, preds, val_df):
    print("\n" + "=" * 65)
    print("CHECK 6: Saving evaluation charts")
    print("=" * 65)

    class_names = list(ID2LABEL.values())
    report      = classification_report(labels, preds, target_names=class_names,
                                        output_dict=True, zero_division=0)

    # ── Figure A: Overall confusion matrix + per-class F1 ─────────────────────
    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    cm_overall = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(confusion_matrix=cm_overall, display_labels=class_names).plot(
        ax=ax1, cmap='Blues', colorbar=False
    )
    ax1.set_title("Overall Confusion Matrix\n(Validation Set)", fontsize=13, pad=10)

    ax2       = fig.add_subplot(gs[1])
    f1_scores = [report[c]['f1-score'] for c in class_names]
    colors    = ['#2196F3' if s >= 0.60 else '#FF5722' for s in f1_scores]
    bars      = ax2.bar(class_names, f1_scores, color=colors, width=0.5, edgecolor='white')
    ax2.axhline(y=0.60, color='gray', linestyle='--', linewidth=1, label='Threshold (0.60)')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Per-Class F1 Score", fontsize=13, pad=10)
    ax2.legend()
    for bar, score in zip(bars, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=11)

    plt.suptitle("Eatsplorer ABSA — Model Evaluation", fontsize=15, y=1.02)
    plt.tight_layout()
    overall_path = os.path.join(OUTPUT_DIR, "evaluation_charts.png")
    plt.savefig(overall_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Overall charts saved      → {overall_path}")

    # ── Figure B: Per-aspect confusion matrices (2×2 grid) ────────────────────
    val_df = val_df.copy()
    val_df['label'] = labels
    val_df['pred']  = preds

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    fig.suptitle("Eatsplorer ABSA — Per-Aspect Confusion Matrices", fontsize=15, y=1.01)

    for ax, aspect in zip(axes.flat, ASPECTS):
        sub = val_df[val_df['aspect'] == aspect]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        sub_labels = sub['label'].tolist()
        sub_preds  = sub['pred'].tolist()

        # Only include label IDs that actually appear in this aspect's subset
        present_ids   = sorted(set(sub_labels) | set(sub_preds))
        present_names = [ID2LABEL[i] for i in present_ids]

        cm_asp = confusion_matrix(sub_labels, sub_preds, labels=present_ids)
        disp   = ConfusionMatrixDisplay(confusion_matrix=cm_asp, display_labels=present_names)
        disp.plot(ax=ax, cmap='Purples', colorbar=False)

        asp_acc = accuracy_score(sub_labels, sub_preds)
        asp_f1  = f1_score(sub_labels, sub_preds, average='macro', zero_division=0)
        ax.set_title(
            f"{aspect.title()}\n"
            f"Acc: {asp_acc:.3f}  |  F1: {asp_f1:.3f}  |  n={len(sub)}",
            fontsize=11, pad=8
        )

    # Hide the unused 6th cell in the 2x3 grid
    axes.flat[5].set_visible(False)

    plt.tight_layout()
    aspect_cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_per_aspect.png")
    plt.savefig(aspect_cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Per-aspect CMs saved      → {aspect_cm_path}")

    # ── Figure C: Per-aspect F1 bar chart ─────────────────────────────────────
    asp_f1_scores = []
    asp_acc_scores = []
    for aspect in ASPECTS:
        sub = val_df[val_df['aspect'] == aspect]
        asp_f1_scores.append(
            f1_score(sub['label'], sub['pred'], average='macro', zero_division=0) if len(sub) else 0
        )
        asp_acc_scores.append(
            accuracy_score(sub['label'], sub['pred']) if len(sub) else 0
        )

    x      = np.arange(len(ASPECTS))
    width  = 0.35
    labels_clean = [a.title() for a in ASPECTS]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, asp_f1_scores,  width, label='F1 Macro',  color='#6A1B9A', edgecolor='white')
    bars2 = ax.bar(x + width/2, asp_acc_scores, width, label='Accuracy',  color='#2E7D32', edgecolor='white')
    ax.axhline(y=0.60, color='gray', linestyle='--', linewidth=1, label='F1 Target (0.60)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_clean, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Aspect F1 and Accuracy (Validation Set)", fontsize=13, pad=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    asp_bar_path = os.path.join(OUTPUT_DIR, "per_aspect_scores.png")
    plt.savefig(asp_bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Per-aspect score chart    → {asp_bar_path}")

# ==========================================
# MAIN
# ==========================================
def main():
    print("\n🔎 EATSPLORER MODEL EVALUATION")
    print("   Checking if your training actually worked...\n")

    model, tokenizer, device = check_model_loads()
    check_sanity(model, tokenizer, device)

    result = check_full_metrics(model, tokenizer, device)
    if result is None or result[0] is None:
        print("\n⚠️  Skipping checks 4-6 (no CSV found).")
        return
    labels, preds, val_df = result

    check_per_class(labels, preds)
    check_per_aspect(labels, preds, val_df)
    check_confusion_and_charts(labels, preds, val_df)

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro')

    print("\n" + "=" * 65)
    print("FINAL VERDICT")
    print("=" * 65)
    if f1 >= 0.70:
        print(f"   🎉 EXCELLENT — F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print("      Your model is performing well. Ready for thesis.")
    elif f1 >= 0.60:
        print(f"   ✅ GOOD — F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print("      Solid results. Check per-class F1 for any weak spots.")
    elif f1 >= 0.50:
        print(f"   ⚠️  ACCEPTABLE — F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print("      Model works but could be improved.")
        print("      Consider: more data for weak classes, or more epochs.")
    else:
        print(f"   ❌ POOR — F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print("      Something went wrong. Check:")
        print("      1. Was training on GPU? (33hr runtime = CPU training)")
        print("      2. Did all epochs complete without errors?")
        print("      3. Look at the sanity test results above for clues.")

    print(f"\n   📁 All charts saved to: ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
