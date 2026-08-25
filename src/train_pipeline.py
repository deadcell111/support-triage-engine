"""Train and evaluate the three-stage ticket pipeline.

Split discipline
----------------
  TRAIN-A  -> fits stage 1
  TRAIN-B  -> stage 1 predicts on it; stage 2 fits on those PREDICTIONS
  TEST     -> held out from everything

Stage 2 must never be fit on stage-1 outputs from stage-1's own training rows:
the intent model is near-perfect on data it memorised, so the gate would learn
to trust a signal that is much noisier in production.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             log_loss, mean_absolute_error, roc_auc_score)
from sklearn.model_selection import train_test_split

from pipeline import TicketPipeline, Config, Decision, TEXT

t0 = time.time()
NEED = ['ticket_id', 'created_at', TEXT, 'auto_resolved', 'sla_breached',
        'issue_type_id', 'category', 'issue_complexity_score', 'priority',
        'product', 'channel', 'region', 'language', 'customer_gender',
        'subscription_type', 'customer_segment', 'customer_age',
        'customer_tenure_months', 'previous_tickets']

d = pd.read_csv('data/ground_truth.csv', usecols=NEED, low_memory=False)
d = d[d.auto_resolved.notna()].reset_index(drop=True)
d['y_solv'] = (d.auto_resolved == 'Yes').astype(int)
d['y_sla'] = (d.sla_breached == 'Yes').astype(int)
print(f"loaded {len(d):,} labelled tickets  {time.time()-t0:.0f}s", flush=True)

teams = pd.read_csv('data/tickets.csv',
                    usecols=['ticket_id', 'assigned_team'], low_memory=False)
d = d.merge(teams, on='ticket_id', how='left')
print(f"joined uncontaminated departments: {d.assigned_team.nunique()} teams", flush=True)

trainval, test = train_test_split(d, test_size=0.2, random_state=42, stratify=d.y_solv)
tr_a, tr_b = train_test_split(trainval, test_size=0.5, random_state=42,
                              stratify=trainval.y_solv)
print(f"TRAIN-A {len(tr_a):,} | TRAIN-B {len(tr_b):,} | TEST {len(test):,}\n", flush=True)

pipe = TicketPipeline(threshold=0.50, draft_floor=0.20)

# ---------------- stage 1 ----------------
print("=" * 62, "\nSTAGE 1  UNDERSTANDING   (text + CRM only)\n" + "=" * 62, flush=True)
pipe.understanding.fit(tr_a)
print(f"  fitted {time.time()-t0:.0f}s", flush=True)
s1_test = pipe.understanding.predict(test)

for col, name in [('issue_type_id', 'intent'), ('category', 'category'),
                  ('priority', 'priority')]:
    acc = accuracy_score(test[col], s1_test[col])
    f1 = f1_score(test[col], s1_test[col], average='macro')
    maj = test[col].value_counts(normalize=True).max()
    print(f"  {name:<12} {test[col].nunique():>3} classes | acc {acc:.4f} "
          f"| macro-F1 {f1:.4f} | majority {maj:.4f}", flush=True)

mae = mean_absolute_error(test.issue_complexity_score, s1_test.issue_complexity_score)
base = mean_absolute_error(test.issue_complexity_score,
                           np.full(len(test), tr_a.issue_complexity_score.median()))
exact = (test.issue_complexity_score.values == s1_test.issue_complexity_score.values).mean()
print(f"  {'complexity':<12}  10 levels | MAE {mae:.3f} | exact {exact:.4f} "
      f"| MAE always-median {base:.3f}", flush=True)

# ---------------- stage 2 ----------------
print("\n" + "=" * 62, "\nSTAGE 2  DECISION   (fed stage-1 PREDICTIONS)\n" + "=" * 62, flush=True)
s1_b = pipe.understanding.predict(tr_b)
pipe.decision.fit(tr_b, s1_b, tr_b.y_solv, tr_b.y_sla)
p_solv, p_sla = pipe.decision.predict(test, s1_test)
print(f"  fitted {time.time()-t0:.0f}s", flush=True)

y = test.y_solv.values
print(f"  solvability  AUC {roc_auc_score(y, p_solv):.4f} "
      f"| ACC {accuracy_score(y, p_solv >= .5):.4f} "
      f"| Brier {brier_score_loss(y, p_solv):.4f}", flush=True)
print(f"  sla risk     AUC {roc_auc_score(test.y_sla, p_sla):.4f} "
      f"| base rate {test.y_sla.mean():.3f}", flush=True)

# --- the pipeline question: how much does stage-1 error cost the gate? ---
true_feats = test[['issue_type_id', 'category', 'priority',
                   'issue_complexity_score']].copy()
true_feats['intent_conf'] = 1.0
true_feats['priority_conf'] = 1.0
p_oracle, _ = pipe.decision.predict(test, true_feats)
print(f"\n  gate on TRUE stage-1 features     AUC {roc_auc_score(y, p_oracle):.4f} "
      f"| ACC {accuracy_score(y, p_oracle >= .5):.4f}", flush=True)
print(f"  gate on PREDICTED features        AUC {roc_auc_score(y, p_solv):.4f} "
      f"| ACC {accuracy_score(y, p_solv >= .5):.4f}", flush=True)
print(f"  cost of cascading stage-1 error   AUC {roc_auc_score(y,p_solv)-roc_auc_score(y,p_oracle):+.4f} "
      f"| ACC {accuracy_score(y,p_solv>=.5)-accuracy_score(y,p_oracle>=.5):+.4f}", flush=True)

# ---------------- stage 3 ----------------
print("\n" + "=" * 62, "\nSTAGE 3  ROUTING   (human-bound only)\n" + "=" * 62, flush=True)
mask = tr_b.assigned_team.notna()
pipe.router.fit(tr_b[mask], s1_b[mask.values], tr_b.assigned_team[mask])
pred_team = pipe.router.predict(test, s1_test)
tm = test.assigned_team.notna()
print(f"  department   {test.assigned_team.nunique()} teams | "
      f"acc {accuracy_score(test.assigned_team[tm], pred_team[tm.values]):.4f} "
      f"| macro-F1 {f1_score(test.assigned_team[tm], pred_team[tm.values], average='macro'):.4f} "
      f"| majority {test.assigned_team.value_counts(normalize=True).max():.4f}", flush=True)

# ---------------- end to end ----------------
print("\n" + "=" * 62, "\nEND TO END\n" + "=" * 62, flush=True)
out = pipe.process(test)
lane = out.lane.value_counts(normalize=True)
print("  lane split:", {k: f"{v:.1%}" for k, v in lane.items()}, flush=True)
auto = out.lane == 'auto_solve'
print(f"  auto-solve precision {(y[auto.values] == 1).mean():.4f} "
      f"on {auto.mean():.1%} of tickets", flush=True)
draft = out.lane == 'ai_draft'
print(f"  ai-draft band carries {draft.mean():.1%}, "
      f"truly solvable within it {(y[draft.values] == 1).mean():.1%}", flush=True)

# ---- what the guardrail costs -------------------------------------------
blk = out.guardrail_blocked.values
would_auto = (p_solv >= 0.50) & blk
print(f"\n  guardrail blocked {blk.sum():,} tickets ({blk.mean():.1%})", flush=True)
print(f"  of those, gate would have auto-solved {would_auto.sum():,} "
      f"({would_auto.mean():.2%} of all tickets)", flush=True)
sacrificed = (would_auto & (y == 1)).sum()
print(f"  genuinely AI-solvable tickets sacrificed for safety: {sacrificed:,} "
      f"({sacrificed/max((y==1).sum(),1):.2%} of all solvable)", flush=True)

pipe.save('models/ticket_pipeline.pkl')
print(f"\nsaved -> ticket_pipeline.pkl   total {time.time()-t0:.0f}s", flush=True)
print("\nsample decisions:")
print(out.head(6)[['ticket_id', 'issue_type_id', 'issue_complexity_score',
                   'p_ai_solvable', 'lane', 'department']].to_string(index=False))
