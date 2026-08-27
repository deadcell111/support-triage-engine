"""
AI-solvable ticket classifier.

Predicts, at intake time, whether a support ticket can be resolved by AI
without escalation to a human.

Model: LightGBM over tabular intake features + an out-of-fold TF-IDF text
score as a stacked feature (method "D": AUC 0.8912 / acc 0.8047 on a
269,451-row held-out split of ground_truth.csv).

Ground truth (`auto_resolved`) and every post-resolution field are excluded
by construction: features are an explicit whitelist, and any leaky column
reaching the feature matrix raises.
"""

from __future__ import annotations
import pickle
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# --- what the model is allowed to see -------------------------------------
# Known at the moment the ticket is created. Nothing here depends on how the
# ticket was handled or how it turned out.
CATEGORICAL = [
    "category", "issue_type_id", "product", "channel", "region",
    "language", "customer_gender", "subscription_type", "customer_segment",
    "priority",
]
NUMERIC = [
    "customer_age", "customer_tenure_months", "previous_tickets",
    "issue_complexity_score",
]
TEXT = "issue_description"
TIMESTAMP = "created_at"

# --- what the model must never see ----------------------------------------
# The label itself, the routing decision, and everything produced during or
# after handling. Any of these in the feature matrix is a leak.
LEAKY = {
    "auto_resolved", "escalated_to_human", "escalated", "escalation_reason",
    "assigned_team", "assigned_to", "resolved_by", "resolution_notes",
    "status", "reopen_count", "first_response_time_hours",
    "resolution_time_hours", "paused_hours", "resolution_wallclock_hours",
    "first_resolved_at", "resolved_at", "closed_at",
    "sla_breached", "sla_breach_margin", "customer_satisfaction_score",
}

LABEL = "auto_resolved"
POSITIVE = "Yes"


@dataclass
class Config:
    """Everything tunable. Defaults reproduce the benchmarked model."""
    threshold: float = 0.50
    use_text: bool = True          # False -> tabular-only (method A)
    categorical: list[str] = field(default_factory=lambda: list(CATEGORICAL))
    numeric: list[str] = field(default_factory=lambda: list(NUMERIC))
    text_column: str = TEXT
    timestamp_column: str = TIMESTAMP
    label_column: str = LABEL
    positive_value: str = POSITIVE
    text_folds: int = 3
    tfidf: dict = field(default_factory=lambda: dict(
        ngram_range=(1, 2), min_df=20, max_features=300_000,
        sublinear_tf=True, strip_accents="unicode",
    ))
    lgbm: dict = field(default_factory=lambda: dict(
        n_estimators=1500, learning_rate=0.05, num_leaves=127,
        min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, n_jobs=-1, verbose=-1,
    ))
    # logloss, not auc: AUC only measures ranking and plateaus within ~30
    # trees, stopping before the probabilities finish calibrating.
    eval_metric: str = "binary_logloss"
    early_stopping_rounds: int = 50
    random_state: int = 42


class AISolvableClassifier:
    def __init__(self, config: Config | None = None, **overrides):
        self.cfg = config or Config()
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise ValueError(f"unknown config option: {k}")
            setattr(self.cfg, k, v)
        self.model_ = None
        self.text_model_ = None
        self.vectorizer_ = None
        self.categories_ = {}
        self.feature_names_ = []
        self.base_rate_ = None

    # -- feature construction ---------------------------------------------
    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        cats = [c for c in cfg.categorical if c in df.columns]
        nums = [c for c in cfg.numeric if c in df.columns]
        X = df[cats + nums].copy()

        ts = cfg.timestamp_column
        if ts in df.columns:
            t = pd.to_datetime(df[ts], errors="coerce")
            X["hour"] = t.dt.hour
            X["dow"] = t.dt.dayofweek
            X["month"] = t.dt.month

        txt = cfg.text_column
        if txt in df.columns:
            s = df[txt].fillna("")
            X["desc_len"] = s.str.len()
            X["desc_words"] = s.str.count(" ") + 1

        leaked = LEAKY & set(X.columns)
        if leaked:
            raise RuntimeError(f"leakage: {sorted(leaked)} reached the features")
        return X

    def _apply_categories(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for c in [c for c in self.cfg.categorical if c in X.columns]:
            if fit:
                self.categories_[c] = pd.CategoricalDtype(
                    sorted(X[c].dropna().astype(str).unique())
                )
            # unseen categories at predict time become NaN, which LightGBM handles
            X[c] = X[c].astype(str).astype(self.categories_[c])
        return X

    def _labels(self, df: pd.DataFrame) -> pd.Series:
        col = df[self.cfg.label_column]
        return (col == self.cfg.positive_value).astype(int)

    # -- training ----------------------------------------------------------
    def fit(self, df: pd.DataFrame, eval_df: pd.DataFrame | None = None):
        cfg = self.cfg
        lab = cfg.label_column
        if lab not in df.columns:
            raise ValueError(f"training data needs the label column {lab!r}")

        # Unlabeled rows are tickets still Open -- outcome unknown, not negative.
        df = df[df[lab].notna()].copy()
        y = self._labels(df)
        self.base_rate_ = float(y.mean())

        X = self._apply_categories(self._engineer(df), fit=True)

        want_text = cfg.use_text and cfg.text_column in df.columns
        if want_text:
            X["text_score"] = self._fit_text(df[cfg.text_column], y)

        self.feature_names_ = list(X.columns)

        fit_kw = {}
        if eval_df is not None:
            ev = eval_df[eval_df[lab].notna()]
            fit_kw = dict(
                eval_X=self.transform(ev), eval_y=self._labels(ev),
                eval_metric=cfg.eval_metric,
                callbacks=[lgb.early_stopping(cfg.early_stopping_rounds,
                                              verbose=False),
                           lgb.log_evaluation(0)],
            )

        self.model_ = lgb.LGBMClassifier(random_state=cfg.random_state, **cfg.lgbm)
        self.model_.fit(X, y, **fit_kw)
        return self

    def _fit_text(self, text: pd.Series, y: pd.Series) -> np.ndarray:
        """Out-of-fold text scores for training; a full-data model for inference.

        OOF is what keeps the stacked feature honest -- a text model that had
        already seen a row's own label would hand the GBM an inflated score.
        """
        cfg = self.cfg
        self.vectorizer_ = TfidfVectorizer(**cfg.tfidf)
        T = self.vectorizer_.fit_transform(text.fillna(""))

        oof = np.zeros(len(y))
        skf = StratifiedKFold(cfg.text_folds, shuffle=True,
                              random_state=cfg.random_state)
        for tr_i, te_i in skf.split(T, y):
            m = LogisticRegression(C=2.0, max_iter=1000, solver="liblinear")
            m.fit(T[tr_i], y.values[tr_i])
            oof[te_i] = m.predict_proba(T[te_i])[:, 1]

        self.text_model_ = LogisticRegression(C=2.0, max_iter=1000,
                                              solver="liblinear").fit(T, y)
        return oof

    # -- inference ---------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._apply_categories(self._engineer(df), fit=False)
        if self.text_model_ is not None:
            col = self.cfg.text_column
            if col in df.columns:
                T = self.vectorizer_.transform(df[col].fillna(""))
                X["text_score"] = self.text_model_.predict_proba(T)[:, 1]
            else:
                # no description available -- fall back to the base rate so the
                # GBM sees a neutral value rather than a missing column
                X["text_score"] = self.base_rate_
        return X.reindex(columns=self.feature_names_)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit() or load() first")
        return self.model_.predict_proba(self.transform(df))[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        thr = self.cfg.threshold if threshold is None else threshold
        return (self.predict_proba(df) >= thr).astype(int)

    def route(self, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        """Per-ticket routing decision plus the score behind it."""
        p = self.predict_proba(df)
        thr = self.cfg.threshold if threshold is None else threshold
        out = pd.DataFrame({
            "ai_solvable_prob": p,
            "ai_solvable": (p >= thr).astype(int),
            "route": np.where(p >= thr, "AI Assistant", "Human"),
        }, index=df.index)
        if "ticket_id" in df.columns:
            out.insert(0, "ticket_id", df["ticket_id"].values)
        return out

    # -- persistence -------------------------------------------------------
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "AISolvableClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
