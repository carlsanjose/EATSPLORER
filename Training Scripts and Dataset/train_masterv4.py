import pandas as pd
import torch
import torch.nn as nn
import emoji
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. THESIS CONFIGURATION
# ==========================================

CSV_FILE   = "training_dataset_revision.csv"
MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
OUTPUT_DIR = "./eatsplorer_finetuned_model"

LABEL2ID = {"N/A": 0, "Negative": 1, "Neutral": 2, "Positive": 3}
ID2LABEL  = {0: "N/A", 1: "Negative", 2: "Neutral", 3: "Positive"}
ASPECTS   = ["food quality", "service", "ambiance", "price value", "overall"]

# ==========================================
# 2. CUDA CHECK
# ==========================================
def verify_cuda():
    print("🔍 Checking GPU...")

    if not torch.cuda.is_available():
        raise SystemExit(
            "\n❌ CUDA NOT FOUND — training aborted.\n"
            "   Possible fixes:\n"
            "   1. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
            "   2. Check that your GPU drivers are up to date\n"
            "   3. Run 'nvidia-smi' — if it fails, reinstall drivers\n"
            "   Training on CPU with this dataset would take 10-20+ hours."
        )

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   ✅ GPU  : {gpu_name}")
    print(f"   ✅ VRAM : {vram_gb:.1f} GB")

    if vram_gb < 3.5:
        print(
            f"   ⚠️  WARNING: Only {vram_gb:.1f}GB VRAM.\n"
            f"      You may hit OOM. Lower max_length=96 or batch_size=1."
        )

    torch.cuda.empty_cache()
    print(f"   ✅ CUDA : {torch.version.cuda}")
    return torch.device("cuda")

# ==========================================
# 3. PREPROCESSING & DATASET
# ==========================================
def preprocess_text(text, rating):
    text = str(text)
    text = emoji.demojize(text)
    return f"[RATING={rating}] {text}"

class EatsplorerDataset(Dataset):
    def __init__(self, encodings, labels, aspects=None):
        self.encodings = encodings
        self.labels    = labels
        self.aspects   = aspects   # kept so callbacks can slice by aspect

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def prepare_data(tokenizer):
    print(f"\n📂 Loading: {CSV_FILE}")

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"'{CSV_FILE}' not found. Run merge_csvs.py first to generate it."
        )

    df = pd.read_csv(CSV_FILE, keep_default_na=False)
    # Normalize casing — CSV has Title Case aspects and a few lowercase sentiments
    df['aspect']    = df['aspect'].str.strip().str.lower()
    df['sentiment'] = df['sentiment'].str.strip().str.capitalize()
    df['sentiment'] = df['sentiment'].replace({'': 'N/A', 'N/a': 'N/A'})
    df = df[df['sentiment'].isin(LABEL2ID.keys())]

    print(f"\n📊 Total samples: {len(df)}")
    print("   Label distribution:")
    for label, count in df['sentiment'].value_counts().items():
        print(f"     {label}: {count} ({count/len(df)*100:.1f}%)")

    df['processed_text'] = df.apply(
        lambda x: preprocess_text(x['review_text'], x['rating']), axis=1
    )
    df['label_id'] = df['sentiment'].map(LABEL2ID).astype(int)

    texts   = df['processed_text'].tolist()
    aspects = df['aspect'].tolist()
    labels  = df['label_id'].tolist()

    # Stratified 70/30 split
    (train_texts, val_texts,
     train_aspects, val_aspects,
     train_labels, val_labels) = train_test_split(
        texts, aspects, labels,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )

    print(f"\n📐 Train: {len(train_labels)} samples | Val: {len(val_labels)} samples")

    train_encodings = tokenizer(
        text=train_aspects,
        text_pair=train_texts,
        truncation=True,
        padding=True,
        max_length=128
    )
    val_encodings = tokenizer(
        text=val_aspects,
        text_pair=val_texts,
        truncation=True,
        padding=True,
        max_length=128
    )

    return (
        EatsplorerDataset(train_encodings, train_labels, train_aspects),
        EatsplorerDataset(val_encodings,   val_labels,   val_aspects),
        train_labels
    )

# ==========================================
# 4. METRICS
# ==========================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        'accuracy':  accuracy_score(labels, preds),
        'f1':        f1_score(labels, preds, average='macro'),
        'precision': precision_score(labels, preds, average='macro', zero_division=0),
        'recall':    recall_score(labels, preds, average='macro', zero_division=0),
    }

# ==========================================
# 5. PER-ASPECT CONFUSION MATRIX HELPER
#    Shared by the callback (per-epoch) and
#    the final post-training plot.
# ==========================================
def plot_per_aspect_confusion_matrices(labels, preds, val_aspects, epoch, save_dir):
    """
    Draws the 2×2 grid of per-aspect confusion matrices exactly like the
    image in the thesis — purple colormap, Acc/F1/n in the title.

    labels      : list[int]  — true label IDs
    preds       : list[int]  — predicted label IDs
    val_aspects : list[str]  — aspect string for each row
    epoch       : int        — used in the filename and suptitle
    save_dir    : str        — directory to write the PNG into
    """
    labels      = np.array(labels)
    preds       = np.array(preds)
    val_aspects = np.array(val_aspects)

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    fig.suptitle(
        f"Eatsplorer ABSA — Per-Aspect Confusion Matrices  [Epoch {epoch}]",
        fontsize=15, y=1.01
    )

    for ax, aspect in zip(axes.flat, ASPECTS):
        mask       = val_aspects == aspect
        sub_labels = labels[mask]
        sub_preds  = preds[mask]

        if len(sub_labels) == 0:
            ax.set_visible(False)
            continue

        # Only include IDs that actually appear in this aspect's subset
        present_ids   = sorted(set(sub_labels.tolist()) | set(sub_preds.tolist()))
        present_names = [ID2LABEL[i] for i in present_ids]

        cm_asp = confusion_matrix(sub_labels, sub_preds, labels=present_ids)
        disp   = ConfusionMatrixDisplay(
            confusion_matrix=cm_asp,
            display_labels=present_names
        )
        disp.plot(ax=ax, cmap='Purples', colorbar=False)

        asp_acc = accuracy_score(sub_labels, sub_preds)
        asp_f1  = f1_score(sub_labels, sub_preds, average='macro', zero_division=0)
        ax.set_title(
            f"{aspect.title()}\n"
            f"Acc: {asp_acc:.3f}  |  F1: {asp_f1:.3f}  |  n={len(sub_labels)}",
            fontsize=11, pad=8
        )

    # Hide the unused 6th cell in the 2x3 grid
    axes.flat[5].set_visible(False)

    plt.tight_layout()
    filename  = f"confusion_matrix_per_aspect_epoch{epoch:02d}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

# ==========================================
# 6. EPOCH HISTORY + PER-ASPECT CM CALLBACK
#    Runs after every eval (= every epoch).
#    Also handles the save-block: epoch 8
#    completes training but is NOT saved.
# ==========================================
class EpochHistoryCallback(TrainerCallback):
    """
    • Records per-epoch train/val metrics for the summary table + curves.
    • After each eval generates the per-aspect confusion matrix PNG.
    • Blocks model saving at epoch 8 (last epoch runs for evaluation only).
    """

    def __init__(self, val_dataset, val_aspects_list, per_aspect_dir):
        self.val_dataset       = val_dataset
        self.val_aspects_list  = val_aspects_list   # list[str] aligned with val set
        self.per_aspect_dir    = per_aspect_dir
        self.epoch_records     = []
        self._train_loss_sum   = 0.0
        self._train_loss_steps = 0

    # ── accumulate step-level train loss ─────────────────────────────────────
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self._train_loss_sum   += logs['loss']
            self._train_loss_steps += 1

    # ── after each epoch evaluation ───────────────────────────────────────────
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics is None:
            return

        epoch = int(round(metrics.get('epoch', len(self.epoch_records) + 1)))

        avg_train_loss = (
            self._train_loss_sum / self._train_loss_steps
            if self._train_loss_steps > 0 else float('nan')
        )

        self.epoch_records.append({
            'epoch':      epoch,
            'train_loss': avg_train_loss,
            'val_loss':   metrics.get('eval_loss',      float('nan')),
            'accuracy':   metrics.get('eval_accuracy',  float('nan')),
            'f1':         metrics.get('eval_f1',        float('nan')),
            'precision':  metrics.get('eval_precision', float('nan')),
            'recall':     metrics.get('eval_recall',    float('nan')),
        })

        # Reset accumulators for next epoch
        self._train_loss_sum   = 0.0
        self._train_loss_steps = 0

        # ── Per-aspect confusion matrix for this epoch ────────────────────────
        if model is not None:
            device = next(model.parameters()).device
            model.eval()

            # Run inference on the full validation set in one batched call
            # We use the trainer's predict pipeline via a small manual loop
            # so we don't need a second Trainer reference here.
            all_preds  = []
            batch_size = 32
            n          = len(self.val_dataset)

            with torch.no_grad():
                for start in range(0, n, batch_size):
                    end   = min(start + batch_size, n)
                    batch = [self.val_dataset[i] for i in range(start, end)]
                    input_ids      = torch.stack([b['input_ids']      for b in batch]).to(device)
                    attention_mask = torch.stack([b['attention_mask']  for b in batch]).to(device)

                    # token_type_ids is optional depending on the model
                    if 'token_type_ids' in batch[0]:
                        token_type_ids = torch.stack(
                            [b['token_type_ids'] for b in batch]
                        ).to(device)
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                        )
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )

                    all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy().tolist())

            true_labels = [self.val_dataset[i]['labels'].item() for i in range(n)]

            os.makedirs(self.per_aspect_dir, exist_ok=True)
            save_path = plot_per_aspect_confusion_matrices(
                true_labels, all_preds,
                self.val_aspects_list,
                epoch,
                self.per_aspect_dir
            )
            print(f"   📊 Per-aspect CM (epoch {epoch}) → {save_path}")

    # ── block saving at epoch 8 ───────────────────────────────────────────────
    def on_save(self, args, state, control, **kwargs):
        """
        Called just before the Trainer writes a checkpoint.
        If the current epoch is 8, cancel the save so no checkpoint
        or model weights are written for that epoch.
        The model still trains through epoch 8 for evaluation purposes.
        """
        current_epoch = int(round(state.epoch)) if state.epoch else 0
        if current_epoch >= 8:
            control.should_save = False
            print(
                f"   ⛔ Epoch {current_epoch}: checkpoint save BLOCKED "
                f"(training runs to epoch 8 for evaluation only — "
                f"best model is from epoch ≤ 7)"
            )
        return control

# ==========================================
# 7. CUSTOM WEIGHTED TRAINER
# ==========================================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        weights = self.class_weights.to(model.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 8. FINAL OVERALL CONFUSION MATRIX
# ==========================================
def plot_overall_confusion_matrix(trainer, val_dataset):
    """Saves the single overall sentiment confusion matrix after training."""
    predictions = trainer.predict(val_dataset)
    preds  = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    cm   = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(ID2LABEL.values())
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title("Eatsplorer ABSA — Overall Confusion Matrix (Validation Set)", pad=12)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix_overall.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   📊 Overall confusion matrix    → {save_path}")

# ==========================================
# 9. TRAINING CURVES
# ==========================================
def plot_training_curves(epoch_records):
    if not epoch_records:
        print("   ⚠️  No epoch records found — skipping training curves.")
        return

    epochs    = [r['epoch']      for r in epoch_records]
    tr_loss   = [r['train_loss'] for r in epoch_records]
    val_loss  = [r['val_loss']   for r in epoch_records]
    accuracy  = [r['accuracy']   for r in epoch_records]
    f1        = [r['f1']         for r in epoch_records]
    precision = [r['precision']  for r in epoch_records]
    recall    = [r['recall']     for r in epoch_records]

    # ── Chart 1: Loss ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, tr_loss,  marker='o', linewidth=2.2, label='Train Loss',      color='#1565C0')
    ax.plot(epochs, val_loss, marker='s', linewidth=2.2, label='Validation Loss', color='#E53935', linestyle='--')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss",  fontsize=12)
    ax.set_title("Training vs Validation Loss per Epoch", fontsize=13, pad=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(OUTPUT_DIR, "training_curves_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"   📈 Loss curve saved            → {loss_path}")

    # ── Chart 2: Metrics ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, accuracy,  marker='o', linewidth=2.2, label='Accuracy',  color='#2E7D32')
    ax.plot(epochs, f1,        marker='s', linewidth=2.2, label='F1 Macro',  color='#6A1B9A')
    ax.plot(epochs, precision, marker='^', linewidth=2.2, label='Precision', color='#E65100', linestyle='--')
    ax.plot(epochs, recall,    marker='v', linewidth=2.2, label='Recall',    color='#00838F', linestyle='--')
    ax.axhline(y=0.60, color='gray', linestyle=':', linewidth=1.2, label='F1 Target (0.60)')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Validation Metrics per Epoch", fontsize=13, pad=10)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    metrics_path = os.path.join(OUTPUT_DIR, "training_curves_metrics.png")
    plt.savefig(metrics_path, dpi=150)
    plt.close()
    print(f"   📈 Metrics curve saved         → {metrics_path}")

# ==========================================
# 10. PER-EPOCH SUMMARY TABLE
# ==========================================
def print_epoch_summary(epoch_records):
    if not epoch_records:
        return

    print("\n" + "=" * 84)
    print("  PER-EPOCH TRAINING SUMMARY")
    print("=" * 84)
    print(
        f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>9}"
        f"  {'Accuracy':>9}  {'F1 Macro':>9}  {'Precision':>10}  {'Recall':>8}  {'':>3}"
    )
    print("  " + "─" * 80)

    best_f1    = -1
    best_epoch = -1
    for r in epoch_records:
        if r['f1'] > best_f1:
            best_f1    = r['f1']
            best_epoch = r['epoch']

    for r in epoch_records:
        star   = "⭐" if r['epoch'] == best_epoch else "  "
        note   = " [no save]" if r['epoch'] >= 8 else ""
        print(
            f"  {r['epoch']:>5}  "
            f"{r['train_loss']:>11.4f}  "
            f"{r['val_loss']:>9.4f}  "
            f"{r['accuracy']:>9.4f}  "
            f"{r['f1']:>9.4f}  "
            f"{r['precision']:>10.4f}  "
            f"{r['recall']:>8.4f}  "
            f"{star}{note}"
        )

    print("  " + "─" * 80)
    print(f"  ⭐ Best epoch: {best_epoch}  |  Best F1: {best_f1:.4f}")
    print("=" * 84)

    summary_df = pd.DataFrame(epoch_records)
    csv_path   = os.path.join(OUTPUT_DIR, "epoch_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  💾 Epoch summary CSV saved     → {csv_path}")

# ==========================================
# 11. TRAINING LOOP
# ==========================================
def train_model():
    print("🚀 EATSPLORER LOCAL TRAINING PIPELINE")

    device    = verify_cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    train_dataset, val_dataset, train_labels_list = prepare_data(tokenizer)
    torch.cuda.empty_cache()

    # Derive class weights
    unique_classes   = np.unique(train_labels_list)
    computed_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_labels_list
    )
    full_weights = np.ones(4)
    for cls, w in zip(unique_classes, computed_weights):
        full_weights[cls] = w

    print(
        f"\n⚖️  Class weights — "
        f"N/A: {full_weights[0]:.3f} | "
        f"Negative: {full_weights[1]:.3f} | "
        f"Neutral: {full_weights[2]:.3f} | "
        f"Positive: {full_weights[3]:.3f}"
    )

    # Directory for per-epoch per-aspect confusion matrices
    per_aspect_dir = os.path.join(OUTPUT_DIR, "per_epoch_aspect_cms")
    os.makedirs(per_aspect_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,
        save_total_limit=2,                          # trains through epoch 8 …
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",          # … but on_save blocks epoch-8 checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        logging_steps=100,
        seed=42,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )

    history_cb = EpochHistoryCallback(
        val_dataset      = val_dataset,
        val_aspects_list = val_dataset.aspects,   # list[str] stored on dataset
        per_aspect_dir   = per_aspect_dir,
    )

    trainer = WeightedTrainer(
        class_weights=full_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=4),
            history_cb,
        ],
    )

    print("\n🔥 STARTING TRAINING...")
    print("   • Epochs 1–7 : train + evaluate + save checkpoint")
    print("   • Epoch 8    : train + evaluate only  (no checkpoint saved)")
    print()
    latest_checkpoint = "/content/drive/MyDrive/Eatsplorer/eatsplorer_finetuned_model/checkpoint-26904"
    print(f"Resuming from: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    torch.cuda.empty_cache()

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n📊 Final evaluation (best model loaded):")
    results = trainer.evaluate()
    print(f"   Accuracy : {results['eval_accuracy']:.4f}")
    print(f"   F1-Score : {results['eval_f1']:.4f}")
    print(f"   Precision: {results['eval_precision']:.4f}")
    print(f"   Recall   : {results['eval_recall']:.4f}")

    # ── Per-epoch summary table ────────────────────────────────────────────────
    print_epoch_summary(history_cb.epoch_records)

    # ── Graphic outputs ────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n📈 Saving final training graphics...")
    plot_training_curves(history_cb.epoch_records)
    plot_overall_confusion_matrix(trainer, val_dataset)

    # Final per-aspect CM using best model
    final_preds_obj = trainer.predict(val_dataset)
    final_preds     = final_preds_obj.predictions.argmax(-1).tolist()
    final_labels    = final_preds_obj.label_ids.tolist()
    best_epoch_num  = max(
        history_cb.epoch_records, key=lambda r: r['f1']
    )['epoch']
    final_asp_path = plot_per_aspect_confusion_matrices(
        final_labels, final_preds,
        val_dataset.aspects,
        epoch=best_epoch_num,
        save_dir=OUTPUT_DIR
    )
    print(f"   📊 Final per-aspect CM (best)  → {final_asp_path}")
    print(f"\n   📁 Per-epoch aspect CMs saved  → {per_aspect_dir}/")

    # ── Save final model ───────────────────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n💾 Model saved to : {OUTPUT_DIR}")
    print("✅ Training complete!")

if __name__ == "__main__":
    train_model()