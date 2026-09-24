"""Baseline: ComplementNB + TF-IDF (word + char n-grams)."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from training.dataset.split import load_splits


def build_pipeline() -> Pipeline:
    """Build NB pipeline with word + char TF-IDF."""

    features = TfidfVectorizer(  # Char TF-IDF turns text into a numeric vector based on character
        analyzer="char_wb",
        # char_wb = build n-grams from characters INSIDE words.
        # So "дурак" -> ["ду", "ур", "ра", "ак", ..., "дур", "ура", ..., ""дурак"].
        ngram_range=(2, 5),
        # ngram_range=(2, 5) = use chunks of 2 to 5 characters.
        # 2-3 chars catch obfuscation and syllables,
        # 4-5 chars catch word roots.
        min_df=2,
        # min_df=2 = ignore n-grams that appear in fewer than 2 documents.
        max_features=50000,
        # max_features=50000 = keep only the 50000 most frequent n-grams.
    )

    clf = CalibratedClassifierCV(ComplementNB(alpha=0.3), cv=3)

    return Pipeline(
        [
            ("features", features),
            ("clf", clf),
        ]
    )


def main():
    # 1. Load splits
    train, val, _ = load_splits()

    # 2. Fit
    pipe = build_pipeline()
    pipe.fit(train["text"], train["label"])

    # 3. Predict
    preds = pipe.predict(val["text"])

    # 4. Metrics
    f1 = f1_score(val["label"], preds)
    precision = precision_score(val["label"], preds)
    recall = recall_score(val["label"], preds)

    print("Baseline: ComplementNB + TF-IDF")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


if __name__ == "__main__":
    main()
