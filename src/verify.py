"""Verification suite — recomputes everything from source and fails loudly.

Nothing here trusts a cached number. Every claim the project makes is
recomputed from the CSVs or from the saved pipeline, and any failure is a
non-zero exit. Writes verify.json.

Modelled on the reference project's phase-4 discipline, with checks specific
to this pipeline: the cascade means the guardrail fires on a *predicted*
issue type, which is a failure mode the reference design does not have.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pipeline import (GUARDRAIL_ISSUE_TYPES, LEAKY, TEXT, TicketPipeline,
                      Decision)

GROUND_TRUTH = "data/ground_truth.csv"
TICKETS = "data/tickets.csv"

checks, fails = [], []


def check(name, ok, detail=""):
    ok = bool(ok)
    checks.append({"check": name, "pass": ok, "detail": str(detail)})
    if not ok:
        fails.append(name)
    print(("PASS" if ok else "FAIL"), "-", name, (f"({detail})" if detail else ""))


# ========================================================================
print("\n--- data integrity ---")
KEY = ["ticket_id", "issue_type_id", "issue_complexity_score", "category",
       "priority", "issue_description"]
g = pd.read_csv(GROUND_TRUTH, usecols=KEY + ["auto_resolved", "escalated_to_human"],
                low_memory=False)
t = pd.read_csv(TICKETS, usecols=KEY + ["assigned_team", "escalated"], low_memory=False)

check("both files have 1,500,000 rows", len(g) == 1_500_000 and len(t) == 1_500_000,
      f"{len(g):,} / {len(t):,}")
check("ticket_id aligned row-for-row", g.ticket_id.equals(t.ticket_id))
same = [c for c in KEY if g[c].equals(t[c])]
check("all shared ticket attributes identical across files",
      len(same) == len(KEY), f"{len(same)}/{len(KEY)} identical")
check("escalation flag identical across files",
      g.escalated_to_human.equals(t.escalated))
check("ground_truth carries the label, tickets does not",
      "auto_resolved" in g.columns and "auto_resolved" not in t.columns)

n_open = g.auto_resolved.isna().sum()
n_res = g.auto_resolved.notna().sum()
check("open + resolved reconciles to total", n_open + n_res == len(g),
      f"{n_open:,} open + {n_res:,} resolved")
check("tickets.csv has no AI Assistant in assigned_team",
      "AI Assistant" not in set(t.assigned_team.dropna()),
      f"{t.assigned_team.nunique()} teams")

# ========================================================================
print("\n--- guardrail ---")
check("guardrail list is 14 issue types", len(GUARDRAIL_ISSUE_TYPES) == 14,
      len(GUARDRAIL_ISSUE_TYPES))
check("no duplicates in guardrail list",
      len(set(GUARDRAIL_ISSUE_TYPES)) == len(GUARDRAIL_ISSUE_TYPES))
unknown = sorted(set(GUARDRAIL_ISSUE_TYPES) - set(g.issue_type_id.unique()))
check("every guardrail type exists in the data", not unknown, unknown)

res = g[g.auto_resolved.notna()]
rate = res.groupby("issue_type_id").auto_resolved.apply(lambda s: (s == "Yes").mean())
green = set(rate[rate >= 0.85].index)
overlap = sorted(green & set(GUARDRAIL_ISSUE_TYPES))
check("guardrail never blocks a >=85% auto-solvable type", not overlap, overlap)

# ========================================================================
print("\n--- pipeline ---")
pipe = TicketPipeline.load("models/ticket_pipeline.pkl")
check("pipeline round-trips through pickle", pipe.decision.solvability is not None)
check("guardrail enabled in saved config", pipe.cfg.guardrail is True)

leaked = LEAKY & set(pipe.decision.features_)
check("no leaky column in the decision feature set", not leaked, sorted(leaked))
check("decision features include stage-1 confidence signals",
      {"intent_conf", "priority_conf"} <= set(pipe.decision.features_))

d = g[g.auto_resolved.notna()].copy()
d["y"] = (d.auto_resolved == "Yes").astype(int)
_, test_idx = train_test_split(np.arange(len(d)), test_size=0.2,
                               random_state=42, stratify=d.y)
full = pd.read_csv(GROUND_TRUTH, low_memory=False)
full = full[full.auto_resolved.notna()].reset_index(drop=True)
test = full.iloc[test_idx].reset_index(drop=True)

INTAKE = ["ticket_id", "created_at", TEXT, "product", "channel", "region",
          "language", "customer_gender", "subscription_type",
          "customer_segment", "customer_age", "customer_tenure_months",
          "previous_tickets"]
out = pipe.process(test[INTAKE])
check("process() runs on intake columns only — no label present",
      "auto_resolved" not in test[INTAKE].columns)
check("every ticket gets exactly one lane",
      len(out) == len(test) and out.lane.notna().all())
check("lane fractions sum to 1",
      abs(out.lane.value_counts(normalize=True).sum() - 1.0) < 1e-9)

# --- cascade-specific: does the guardrail survive predicted issue types? --
pred_block = out.issue_type_id.isin(GUARDRAIL_ISSUE_TYPES).to_numpy()
true_block = test.issue_type_id.isin(GUARDRAIL_ISSUE_TYPES).to_numpy()
agree = (pred_block == true_block).mean()
missed = int((true_block & ~pred_block).sum())
check("guardrail fires identically on predicted vs true issue type",
      agree == 1.0, f"agreement {agree:.6f}, missed {missed}")
check("no guardrail-blocked ticket reaches an AI lane",
      not out.loc[pred_block, "lane"].isin(["auto_solve", "ai_draft"]).any())

y = (test.auto_resolved == "Yes").astype(int).to_numpy()
p = out.p_ai_solvable.to_numpy()
sacrificed = int(((p >= 0.50) & pred_block & (y == 1)).sum())
check("guardrail safety cost stays under 3% of solvable tickets",
      sacrificed / max((y == 1).sum(), 1) < 0.03,
      f"{sacrificed:,} sacrificed = {sacrificed/max((y==1).sum(),1):.2%}")

# ========================================================================
print("\n--- model sanity ---")
auc = roc_auc_score(y, p)
check("end-to-end gate AUC in the expected 0.87-0.90 band", 0.87 <= auc <= 0.90,
      f"{auc:.4f}")
qs = pd.qcut(p, 10, duplicates="drop")
cal = pd.DataFrame({"p": p, "y": y}).groupby(qs, observed=True).agg(
    pred=("p", "mean"), actual=("y", "mean"))
gap = (cal.pred - cal.actual).abs().max()
check("calibration gap under 0.05 in every decile", gap < 0.05, f"max gap {gap:.4f}")

auto = out.lane == "auto_solve"
prec = (y[auto.to_numpy()] == 1).mean()
check("auto-solve precision above 0.90", prec > 0.90, f"{prec:.4f}")
check("auto-solve covers 25-40% of tickets", 0.25 <= auto.mean() <= 0.40,
      f"{auto.mean():.1%}")

# --- ablation: a leak would collapse AUC when complexity is neutralised ---
s1 = pipe.understanding.predict(test)
s1_flat = s1.copy()
s1_flat["issue_complexity_score"] = 5
X = pipe.decision.transform(test, s1_flat)
auc_flat = roc_auc_score(y, pipe.decision.solvability.predict_proba(X)[:, 1])
check("neutralising complexity degrades AUC but does not collapse it",
      0.80 < auc_flat < auc, f"{auc:.4f} -> {auc_flat:.4f}")

# ========================================================================
summary = {"n_checks": len(checks), "n_fail": len(fails), "failed": fails,
           "checks": checks}
json.dump(summary, open("reports/verify.json", "w"), indent=1)
print(f"\n{'='*60}\n{len(checks)-len(fails)}/{len(checks)} passed"
      f"{'' if not fails else '  FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
