"""Fine-tune rubert-tiny2 on toxic comments."""

import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from training.dataset.split import load_splits

MODEL_NAME = "cointegrated/rubert-tiny2"
MAX_LENGTH = 128
OUTPUT_DIR = "artifacts/rubert-tiny2"


def get_device():
    """Pick GPU if available, else CPU with a visible banner."""
    green = "\033[92m"
    yellow = "\033[93m"
    reset = "\033[0m"
    bold = "\033[1m"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            f"""
{bold}{green}============================================================{reset}
  [GPU] TRAINING ON GPU
  Device:  {torch.cuda.get_device_name(0)}
  CUDA:    {torch.version.cuda}
  Capability: {torch.cuda.get_device_capability(0)}
{bold}{green}============================================================{reset}
"""
        )
    else:
        device = torch.device("cpu")
        print(
            f"""
{bold}{yellow}============================================================{reset}
  [CPU] TRAINING ON CPU (GPU not available)
  Install CUDA torch for faster training:
    pip install torch --index-url https://download.pytorch.org/whl/cu118
{bold}{yellow}============================================================{reset}
"""
        )
    return device


def build_model_and_tokenizer():
    """Load rubert-tiny with classification head."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # toxic/not toxic
    )
    return model, tokenizer


def to_hf_dataset(df, tokenizer):
    """Convert DataFrame to HuggingFace Dataset with tokenized inputs."""
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )

    return Dataset.from_dict(
        {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    )


def compute_metrics(pred):
    """Compute F1, precision, recall for Trainer logging."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(axis=-1)
    return {
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


def main():
    # 1. Load splits
    train, val, _ = load_splits()

    # 2. Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer()

    # 3. Move model to GPU/CPU
    device = get_device()
    model.to(device)

    # 4. Prepare datasets
    train_dataset = to_hf_dataset(train, tokenizer)
    val_dataset = to_hf_dataset(val, tokenizer)

    # 5. Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Save model and tokenizer to artifacts/rubert-tiny2
    trainer.save_model("artifacts/rubert-tiny2")
    tokenizer.save_pretrained("artifacts/rubert-tiny2")
    print("Model and tokenizer saved to artifacts/rubert-tiny2")

    # 6. Predict
    preds = trainer.predict(val_dataset)
    predicted_labels = preds.predictions.argmax(axis=-1)

    # 7. Metrics
    f1 = f1_score(val["label"], predicted_labels)
    precision = precision_score(val["label"], predicted_labels)
    recall = recall_score(val["label"], predicted_labels)

    print("Fine-tune: rubert-tiny2")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print("\n--- Comparison with NB baseline ---")
    print("NB F1:        0.8670")
    print(f"Rubert F1:    {f1:.4f}")
    print(f"Improvement:  {f1 - 0.8670:+.4f}")


if __name__ == "__main__":
    main()
