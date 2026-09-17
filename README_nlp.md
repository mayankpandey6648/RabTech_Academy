# Deep Learning Text Classifier — Sentiment Analysis (PyTorch)

A multi-layer neural network sentiment classifier: TF-IDF vectorization, a `nn.Sequential` MLP
with Dropout + BatchNorm, early stopping on validation loss, training-curve plots, and inference
with confidence scores on unseen sentences.

## Contents
- `nlp_text_classifier.ipynb` — full notebook: data, split, TF-IDF, model, training loop with
  early stopping, loss/accuracy curves, test evaluation, inference on unseen samples
- `sentiment_reviews.csv` — training data (860 labeled review-style sentences)
- `text_classifier.pt` — trained model weights (`state_dict`, load with `torch.load(...)`)
- `tfidf_vectorizer.joblib` — fitted TF-IDF vectorizer (load with `joblib.load(...)`)
- `requirements.txt` — Python dependencies

## Architecture
```
Linear(vocab_size → 256) → BatchNorm1d → ReLU → Dropout(0.4)
Linear(256 → 64)         → BatchNorm1d → ReLU → Dropout(0.4)
Linear(64 → 1)                                              # logit, BCEWithLogitsLoss
```
Trained with Adam, early stopping (patience=6 epochs on validation loss), best-epoch weights
restored before final evaluation.

## Result
Test accuracy: 1.000 (129 held-out examples) — see the notebook's note on why this templated
dataset makes the task easier than real-world text, and §8 for a more honest generalization
check on 5 hand-written, non-templated sentences (4/5 correct with well-calibrated confidence,
including a low-confidence call on a genuinely ambiguous sentence).

## Usage
```python
import torch, joblib
import torch.nn as nn

vectorizer = joblib.load('tfidf_vectorizer.joblib')

class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden1=256, hidden2=64, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

model = TextClassifier(input_dim=len(vectorizer.vocabulary_))
model.load_state_dict(torch.load('text_classifier.pt'))
model.eval()

X = vectorizer.transform(["This was amazing, worth every penny."]).toarray().astype('float32')
prob = torch.sigmoid(model(torch.tensor(X))).item()
print("positive" if prob > 0.5 else "negative", prob)
```

## Swapping in real data
Replace `sentiment_reviews.csv` with any two-column `text,label` CSV (e.g. IMDB reviews, a spam/ham
corpus) — the rest of the notebook runs unchanged.
