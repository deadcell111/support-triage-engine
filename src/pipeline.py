"""
Ticket triage pipeline.

Three stages, each seeing only what would exist at that point in a real system:

  STAGE 1  UNDERSTANDING   text + CRM fields  ->  intent, complexity, priority
  STAGE 2  DECISION        stage-1 PREDICTIONS ->  solvability, SLA risk
  STAGE 3  ROUTING         human-bound tickets ->  department

The point of the split is that stage 2 consumes stage 1's *predictions*, never
the ground-truth columns. A ticket arriving in production has a description and
a customer record -- it does not arrive with `issue_type_id` already filled in.
Measuring the gate under predicted features rather than true ones is the whole
reason this is a pipeline and not four independent models.

Stage 3 trains on tickets.csv: in ground_truth.csv the
`assigned_team` column is contaminated by 715k rows reading "AI Assistant",
which is not a department. The human-handled file carries the uncontaminated
routing for the same tickets.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDClassifier

# ---- what a ticket actually arrives with ---------------------------------
TEXT = "issue_description"
INTAKE_CAT = ["product", "channel", "region", "language", "customer_gender",
              "subscription_type", "customer_segment"]
INTAKE_NUM = ["customer_age", "customer_tenure_months", "previous_tickets"]

# ---- what stage 1 infers (NOT available at intake) -----------------------
INFERRED = ["issue_type_id", "category", "issue_complexity_score", "priority"]

# ---- deterministic force-escalate list -----------------------------------
# Issue types an AI must never close on its own, regardless of how confident
# the gate is. These are money disputes, account compromise, and legal data
# requests: the cost of being wrong is not measured in handling hours, and a
# model good enough on average is not a defence when it is wrong on one of
# these. Taken from the reference project's Phase-2 decision.
GUARDRAIL_ISSUE_TYPES = [
    # Security -- compromise, breach, phishing, legal data requests
    "security_compromised_account",
    "security_data_breach_concern",
    "security_phishing_report",
    "security_gdpr_data_request",
    "security_delete_my_data",
    # Refunds -- money disputes and appeals (not routine status/policy lookups)
    "refunds_chargeback_question",
    "refunds_refund_denied_appeal",
    "refunds_wrong_refund_amount",
    "refunds_duplicate_refund",
    "refunds_partial_refund",
    # Billing -- disputes and unauthorised or failed charges
    "billing_billing_dispute",
    "billing_payment_deducted_failed",
    "billing_double_charge",
    "billing_invoice_wrong_amount",
]

# ---- never features anywhere ---------------------------------------------
LEAKY = {
    "auto_resolved", "escalated_to_human", "escalated", "escalation_reason",
    "assigned_team", "assigned_to", "resolved_by", "resolution_notes",
    "status", "reopen_count", "first_response_time_hours",
    "resolution_time_hours", "paused_hours", "resolution_wallclock_hours",
    "first_resolved_at", "resolved_at", "closed_at",
    "sla_breached", "sla_breach_margin", "customer_satisfaction_score",
}


@dataclass
class Config:
    threshold: float = 0.50          # auto-solve above this
    draft_floor: float = 0.20        # AI-drafts between floor and threshold
    guardrail: bool = True           # force-escalate GUARDRAIL_ISSUE_TYPES
    tfidf: dict = field(default_factory=lambda: dict(
        ngram_range=(1, 2), min_df=10, sublinear_tf=True,
        strip_accents="unicode", max_features=200_000))
    sgd: dict = field(default_factory=lambda: dict(
        loss="log_loss", alpha=1e-6, max_iter=15, tol=1e-3,
        n_jobs=-1, random_state=42))
    lgbm: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.05, num_leaves=127,
        min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=42))


def _check_leak(cols):
    bad = LEAKY & set(cols)
    if bad:
        raise RuntimeError(f"leakage: {sorted(bad)} reached the features")


# =========================================================================
# STAGE 1 — understanding
# =========================================================================
class Understanding:
    """Text + CRM -> intent, category, complexity, priority."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vec = None
        self.intent = self.category = self.priority = self.complexity = None

    def _matrix(self, df):
        T = self.vec.transform(df[TEXT].fillna(""))
        num = df[INTAKE_NUM].fillna(-1).to_numpy(np.float32)
        return sp.hstack([T, sp.csr_matrix(num)]).tocsr()

    def fit(self, df):
        _check_leak(INTAKE_CAT + INTAKE_NUM + [TEXT])
        self.vec = TfidfVectorizer(**self.cfg.tfidf)
        self.vec.fit(df[TEXT].fillna(""))
        X = self._matrix(df)
        self.intent = SGDClassifier(**self.cfg.sgd).fit(X, df.issue_type_id)
        self.category = SGDClassifier(**self.cfg.sgd).fit(X, df.category)
        self.priority = SGDClassifier(**self.cfg.sgd).fit(X, df.priority)
        self.complexity = Ridge(alpha=1.0).fit(X, df.issue_complexity_score)
        return self

    def predict(self, df) -> pd.DataFrame:
        X = self._matrix(df)
        cx = np.clip(np.rint(self.complexity.predict(X)), 1, 10)
        out = pd.DataFrame({
            "issue_type_id": self.intent.predict(X),
            "category": self.category.predict(X),
            "priority": self.priority.predict(X),
            "issue_complexity_score": cx.astype(int),
            # confidence signals -- a low-margin intent is itself informative
            "intent_conf": self.intent.predict_proba(X).max(axis=1),
            "priority_conf": self.priority.predict_proba(X).max(axis=1),
        }, index=df.index)
        return out


# =========================================================================
# STAGE 2 — decision
# =========================================================================
class Decision:
    """Stage-1 predictions + intake -> P(ai solvable), P(sla breach)."""

    CAT = INTAKE_CAT + ["issue_type_id", "category", "priority"]
    NUM = INTAKE_NUM + ["issue_complexity_score", "intent_conf",
                        "priority_conf", "hour", "dow", "month",
                        "desc_len", "desc_words"]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.solvability = self.sla_risk = None
        self.categories_ = {}
        self.features_ = []

    def _features(self, df, stage1):
        X = pd.concat([df[INTAKE_CAT + INTAKE_NUM].reset_index(drop=True),
                       stage1.reset_index(drop=True)], axis=1)
        if "created_at" in df.columns:
            t = pd.to_datetime(df.created_at, errors="coerce").reset_index(drop=True)
            X["hour"], X["dow"], X["month"] = t.dt.hour, t.dt.dayofweek, t.dt.month
        s = df[TEXT].fillna("").reset_index(drop=True)
        X["desc_len"], X["desc_words"] = s.str.len(), s.str.count(" ") + 1
        _check_leak(X.columns)
        return X

    def _cast(self, X, fit):
        for c in self.CAT:
            if fit:
                self.categories_[c] = pd.CategoricalDtype(
                    sorted(X[c].dropna().astype(str).unique()))
            X[c] = X[c].astype(str).astype(self.categories_[c])
        return X

    def fit(self, df, stage1, y_solv, y_sla):
        X = self._cast(self._features(df, stage1), fit=True)
        self.features_ = list(X.columns)
        # 121 trees was the early-stopping optimum on this data; fixed here so
        # the pipeline needs no eval set threaded through every stage.
        self.solvability = lgb.LGBMClassifier(**self.cfg.lgbm).fit(X, y_solv)
        self.sla_risk = lgb.LGBMClassifier(**self.cfg.lgbm).fit(X, y_sla)
        return self

    def transform(self, df, stage1):
        return self._cast(self._features(df, stage1), fit=False)\
                   .reindex(columns=self.features_)

    def predict(self, df, stage1):
        X = self.transform(df, stage1)
        return (self.solvability.predict_proba(X)[:, 1],
                self.sla_risk.predict_proba(X)[:, 1])


# =========================================================================
# STAGE 3 — routing
# =========================================================================
class Router:
    """Human-bound tickets -> department. Trained on the human-handled file."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.classes_ = None
        self.categories_ = {}
        self.features_ = []

    def fit(self, df, stage1, teams):
        dec = Decision(self.cfg)
        X = dec._features(df, stage1)
        for c in Decision.CAT:
            self.categories_[c] = pd.CategoricalDtype(
                sorted(X[c].dropna().astype(str).unique()))
            X[c] = X[c].astype(str).astype(self.categories_[c])
        self.features_ = list(X.columns)
        self.model = lgb.LGBMClassifier(**{**self.cfg.lgbm, "n_estimators": 400})
        self.model.fit(X, teams)
        self.classes_ = self.model.classes_
        return self

    def predict(self, df, stage1):
        X = Decision(self.cfg)._features(df, stage1)
        for c, dt in self.categories_.items():
            X[c] = X[c].astype(str).astype(dt)
        X = X.reindex(columns=self.features_)
        return self.model.predict(X)


# =========================================================================
# Orchestrator
# =========================================================================
class TicketPipeline:
    def __init__(self, config: Config | None = None, **over):
        self.cfg = config or Config()
        for k, v in over.items():
            setattr(self.cfg, k, v)
        self.understanding = Understanding(self.cfg)
        self.decision = Decision(self.cfg)
        self.router = Router(self.cfg)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full intake -> decision record. Uses only intake-time inputs."""
        s1 = self.understanding.predict(df)
        p_solv, p_sla = self.decision.predict(df, s1)

        lane = np.where(p_solv >= self.cfg.threshold, "auto_solve",
               np.where(p_solv >= self.cfg.draft_floor, "ai_draft", "human"))

        # The guardrail runs last and overrides the gate unconditionally.
        # Note it fires on the *predicted* issue type -- this pipeline infers
        # that field, so the guardrail inherits stage-1 error. verify.py
        # checks predicted-vs-true firing explicitly.
        blocked = np.zeros(len(df), bool)
        if self.cfg.guardrail:
            blocked = s1.issue_type_id.isin(GUARDRAIL_ISSUE_TYPES).to_numpy()
            lane = np.where(blocked, "guardrail", lane)

        out = pd.concat([s1.reset_index(drop=True)], axis=1)
        out["p_ai_solvable"] = p_solv
        out["p_sla_breach"] = p_sla
        out["guardrail_blocked"] = blocked
        out["lane"] = lane

        needs_team = lane != "auto_solve"
        out["department"] = None
        if needs_team.any():
            sub = df.reset_index(drop=True)[needs_team]
            out.loc[needs_team, "department"] = self.router.predict(
                sub, s1.reset_index(drop=True)[needs_team])
        if "ticket_id" in df.columns:
            out.insert(0, "ticket_id", df.ticket_id.values)
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "TicketPipeline":
        with open(path, "rb") as f:
            return pickle.load(f)
