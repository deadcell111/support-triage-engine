"""Cost-optimal routing threshold and FTE impact.

Effort model
------------
`resolution_time_hours` is not literal agent labour (12h x 1.35M tickets is
impossible for 44 people). It is used as a RELATIVE effort proxy, calibrated
so that the no-AI world equals the 44 FTE the generator actually staffed in
tickets.csv.

Per-ticket human effort, by complexity c:
  routed to AI, AI solves        -> ~0 human hours (AI compute only)
  routed to AI, AI fails         -> ai_fail_h(c)   (wasted attempt + human)
  routed to human                -> human_h(c)

The counterfactual timing columns in the human-handled file are copies of the
AI file, so that file is used ONLY to anchor headcount (44), never for effort.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from ai_solvable import AISolvableClassifier, CATEGORICAL, NUMERIC, TEXT, TIMESTAMP, LABEL

BASELINE_FTE   = 44        # workers in the no-AI dataset
FTE_COST       = 55_000    # fully loaded annual cost, USD  (input assumption)
AI_COST_TICKET = 0.10      # inference + retrieval per AI attempt, USD (input)
DRAFT_SAVING   = 0.70      # human effort removed when AI drafts & human approves

# ---- load + score held-out tickets ---------------------------------------
cols = ['ticket_id', TIMESTAMP, TEXT, LABEL, 'escalated_to_human',
        'resolution_time_hours', 'customer_satisfaction_score'] + CATEGORICAL + NUMERIC
d = pd.read_csv('data/ground_truth.csv', usecols=cols, low_memory=False)
d = d[d[LABEL].notna()]
_, te = train_test_split(d, test_size=0.2, random_state=42,
                         stratify=(d[LABEL] == 'Yes'))

clf = AISolvableClassifier.load('models/ai_solvable_model.pkl')
te = te.copy()
te['p'] = clf.predict_proba(te)
te['y'] = (te[LABEL] == 'Yes').astype(int)

# ---- effort curves by complexity, from observed outcomes -----------------
d['grp'] = np.where(d[LABEL] == 'Yes', 'ai_ok',
           np.where(d.escalated_to_human == 'Yes', 'ai_fail', 'human'))
eff = d.pivot_table(index='issue_complexity_score', columns='grp',
                    values='resolution_time_hours', aggfunc='mean')
csat = d.pivot_table(index='issue_complexity_score', columns='grp',
                     values='customer_satisfaction_score', aggfunc='mean')
human_h = eff['human']; fail_h = eff['ai_fail']

c = te.issue_complexity_score.values
te['h_human'] = human_h.reindex(c).values      # cost if routed to a human
te['h_fail']  = fail_h.reindex(c).values       # cost if AI tries and fails
te['csat_human'] = csat['human'].reindex(c).values
te['csat_fail']  = csat['ai_fail'].reindex(c).values
te['csat_ok']    = csat['ai_ok'].reindex(c).values

# ---- calibrate effort -> FTE --------------------------------------------
baseline_hours = te.h_human.sum()              # no-AI world: every ticket human
hours_per_fte  = baseline_hours / BASELINE_FTE
scale = len(d) / len(te)                       # test set -> full volume
print(f"test tickets {len(te):,}  (= {1/ (1/scale):.0f}x -> full {len(d):,})")
print(f"baseline (no AI): {BASELINE_FTE} FTE, {baseline_hours:,.0f} effort-hours\n")

def evaluate(to_ai, draft=None):
    """to_ai: bool mask routed to autonomous AI. draft: bool mask AI-drafted."""
    y, hh, hf = te.y.values, te.h_human.values, te.h_fail.values
    hours = np.where(to_ai, np.where(y == 1, 0.0, hf), hh)
    if draft is not None:
        hours = np.where(draft & ~to_ai, hh * (1 - DRAFT_SAVING), hours)
    cs = np.where(to_ai, np.where(y == 1, te.csat_ok, te.csat_fail), te.csat_human)
    fte = hours.sum() / hours_per_fte
    ai_attempts = to_ai.sum() + (0 if draft is None else draft.sum())
    return dict(
        automated=to_ai.mean(),
        drafted=0.0 if draft is None else draft.mean(),
        precision=(y[to_ai] == 1).mean() if to_ai.sum() else np.nan,
        hours=hours.sum(), fte=fte, fte_saved=BASELINE_FTE - fte,
        csat=cs.mean(),
        ai_usd=ai_attempts * scale * AI_COST_TICKET,
    )

# ---- 1. global threshold sweep ------------------------------------------
print("=== global threshold sweep ===")
rows = []
for t in np.arange(0.30, 0.96, 0.05):
    r = evaluate(te.p.values >= t); r['thr'] = t; rows.append(r)
g = pd.DataFrame(rows).set_index('thr')
print(g[['automated','precision','fte','fte_saved','csat']].round(3).to_string())
best_t = g.fte.idxmin()
print(f"\ncost-optimal global threshold: {best_t:.2f}  "
      f"-> {g.loc[best_t,'fte']:.1f} FTE  (saves {g.loc[best_t,'fte_saved']:.1f})")

# ---- 2. per-complexity threshold ----------------------------------------
# route to AI iff (1-p)*fail_h(c) < human_h(c)  =>  p > 1 - human_h/fail_h
opt = (1 - human_h / fail_h).rename('thr*')
print("\n=== cost-optimal threshold BY COMPLEXITY ===")
print(pd.concat([human_h.rename('human_h'), fail_h.rename('fail_h'),
                 opt.round(3)], axis=1).round(2).to_string())

thr_c = opt.reindex(c).values
r_dyn = evaluate(te.p.values >= thr_c)
print(f"\nper-complexity rule: automate {r_dyn['automated']:.1%} "
      f"@ precision {r_dyn['precision']:.3f} -> {r_dyn['fte']:.1f} FTE "
      f"(saves {r_dyn['fte_saved']:.1f})")

# ---- 3. add the AI-draft tier -------------------------------------------
print("\n=== + AI-draft/human-approve tier (draft band below thr*) ===")
for lo in [0.20, 0.30, 0.40]:
    to_ai = te.p.values >= thr_c
    draft = (te.p.values >= lo) & ~to_ai
    r = evaluate(to_ai, draft)
    print(f"  draft band [{lo:.2f}, thr*): auto {r['automated']:.1%} + "
          f"draft {r['drafted']:.1%} -> {r['fte']:.1f} FTE "
          f"(saves {r['fte_saved']:.1f}, ${r['fte_saved']*FTE_COST:,.0f}/yr)")

# ---- 4. bounds ----------------------------------------------------------
oracle = evaluate(te.y.values == 1)
naive  = evaluate(te.p.values >= 0.50)
print(f"\n=== bounds ===")
print(f"  perfect classifier (oracle): {oracle['fte']:.1f} FTE "
      f"(saves {oracle['fte_saved']:.1f})")
print(f"  generator's actual staffing: 32.0 FTE (saves 12.0)")
print(f"  our model @ 0.50          : {naive['fte']:.1f} FTE "
      f"(saves {naive['fte_saved']:.1f})")
print(f"  our model @ per-complexity: {r_dyn['fte']:.1f} FTE "
      f"(saves {r_dyn['fte_saved']:.1f})")
print(f"\n  AI inference cost at per-complexity rule: "
      f"${r_dyn['ai_usd']:,.0f}/period vs "
      f"${r_dyn['fte_saved']*FTE_COST:,.0f} labour saved")
