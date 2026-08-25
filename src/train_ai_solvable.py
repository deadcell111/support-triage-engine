"""Train and evaluate the AI-solvable classifier on ground_truth.csv."""

import time

import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix,
                             log_loss, roc_auc_score)
from sklearn.model_selection import train_test_split

from ai_solvable import CATEGORICAL, LABEL, NUMERIC, TEXT, TIMESTAMP, \
    AISolvableClassifier

DATA = "data/ground_truth.csv"
THRESHOLD = 0.50

t0 = time.time()
usecols = ["ticket_id", TIMESTAMP, TEXT, LABEL] + CATEGORICAL + NUMERIC
df = pd.read_csv(DATA, usecols=usecols, low_memory=False)
print(f"loaded {len(df):,} rows in {time.time()-t0:.1f}s")

labeled = df[df[LABEL].notna()]
print(f"labeled {len(labeled):,}  "
      f"(dropped {len(df)-len(labeled):,} still-Open tickets)")

train, test = train_test_split(
    labeled, test_size=0.2, random_state=42,
    stratify=(labeled[LABEL] == "Yes"),
)
print(f"train {len(train):,}  test {len(test):,}")

clf = AISolvableClassifier(threshold=THRESHOLD)
clf.fit(train, eval_df=test)
print(f"trained in {time.time()-t0:.1f}s  "
      f"({clf.model_.best_iteration_ or clf.cfg.lgbm['n_estimators']} trees, "
      f"{len(clf.feature_names_)} features)")

y = (test[LABEL] == "Yes").astype(int)
p = clf.predict_proba(test)
yhat = (p >= THRESHOLD).astype(int)

print(f"\n=== held-out ({len(test):,} tickets, threshold {THRESHOLD}) ===")
print(f"  AUC      {roc_auc_score(y, p):.4f}")
print(f"  Accuracy {accuracy_score(y, yhat):.4f}")
print(f"  LogLoss  {log_loss(y, p):.4f}   Brier {brier_score_loss(y, p):.4f}")
print(classification_report(y, yhat, target_names=["human", "ai_solvable"],
                            digits=3))

tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
n = len(y)
print(f"  routed to AI   {(tp+fp)/n:6.1%}  of which {tp/(tp+fp):.1%} correct")
print(f"  misrouted      {fp/n:6.1%}  (sent to AI, needed a human)")
print(f"  missed         {fn/n:6.1%}  (kept human, AI could have solved)")

clf.save("models/ai_solvable_model.pkl")
print("\nsaved -> ai_solvable_model.pkl")

# round-trip check: reload and route unseen tickets
reloaded = AISolvableClassifier.load("models/ai_solvable_model.pkl")
print(reloaded.route(test.head(5)).to_string(index=False))
