# ai-solvable

**Should an AI answer this support ticket?** A three-stage triage pipeline over 1.5M tickets —
and the finding that a failed AI attempt costs **+13.9 hours**, more than a correct automation saves.

📊 **[Business case →](https://claude.ai/code/artifact/270c52b9-d758-496e-aad3-640cf9b419eb)**  ·  🧪 **[Pilot protocol →](https://claude.ai/code/artifact/693ea9cd-385c-455d-bc4d-4abcc9ac53e2)**

---

## The short version

A support desk of 44 agents can run on 34. But the naive read of the data — *"47.7% of tickets
resolve themselves, so automate 47.7%"* — is wrong, and expensively so.

| Path | Tickets | Resolution | CSAT | SLA breach |
|---|---:|---:|---:|---:|
| AI resolved it | 642,710 | 1.71 h | 4.21 | 0.1% |
| Straight to a human | 293,760 | 12.34 h | 3.54 | 7.7% |
| **AI tried, then escalated** | 410,785 | **30.46 h** | **2.81** | **28.2%** |

A failed AI attempt takes **2.5× longer** than never letting it try. The gap holds at every one
of the ten complexity levels — it is not selection bias — and is widest in relative terms on the
*easiest* tickets (4.2× at complexity 1).

That inverts the objective. The goal is not to automate as many tickets as possible; it is to
automate only the ones the model is genuinely confident about, because **each mistake costs more
than roughly one correct automation saves.**

---

## About the data

**I generated this dataset myself.** That is disclosed up front because it changes what the
results license, and a reader who discovers it later is entitled to discount everything.

**What it means:** every finding below is a property of a simulation I built, not a discovery
about real support desks. The +13.9h penalty is structure the analysis *recovers* from the
generator — it is not evidence that real AI deflection behaves this way.

**Why it still strengthens the project:** the generator was designed with a deliberate
experimental property that most public datasets lack — a **matched counterfactual pair.**

| File | Rows | What it is |
|---|---:|---|
| `data/ground_truth.csv` | 1,500,000 | Routed with an AI assistant. Carries `auto_resolved`, the label. Needs **32 human agents**. |
| `data/tickets.csv` | 1,500,000 | The same tickets, same ordering, every attribute identical — routed to people only. Needs **44 agents**. |

Two files, one ticket universe, differing in exactly three columns: `assigned_team`,
`assigned_to`, `resolved_by`. That design is what makes the headcount question *answerable*
rather than hypothetical — the counterfactual is measured, not assumed. It also gives the
department router an uncontaminated training target, since `assigned_team` in the first file
reads `"AI Assistant"` for 715,653 rows, which is not a department.

**Known artefacts of synthetic generation**, stated so nobody has to find them:

- Descriptions are template-generated and **one-to-one with `issue_type_id`**, so intent
  classification scores a meaningless 100%. Real intent models run 70–85%.
- `resolution_notes` has only **10 distinct values** across 1.4M rows, and they are internal
  summaries rather than customer replies — so no response generator can be trained from this
  data. Reply generation is deliberately out of scope.
- The label is stochastic, not deterministic: **no issue type resolves 0% or 100%** of the time.
  That puts a real ~80% ceiling on any classifier, which is a feature — it makes the accuracy
  question boring and the economics question interesting.
- The generator is stationary. Trained on 2022–23 and tested on 2024, AUC moves by 0.0003 —
  there is no drift to detect, so this data cannot rehearse the one failure mode that matters
  most in production.

---

## Repository layout

```
.
├── src/
│   ├── pipeline.py            three-stage pipeline + guardrail + orchestrator
│   ├── train_pipeline.py      trains all 7 models, prints the full scorecard
│   ├── verify.py              26 checks, recomputed from source, non-zero exit on failure
│   ├── cost_model.py          accuracy -> headcount -> money, with sensitivity
│   ├── eda.py                 full aggregate pass -> reports/eda.json
│   ├── ai_solvable.py         the standalone single-model gate (the first iteration)
│   └── train_ai_solvable.py
├── data/
│   ├── sample_ground_truth.csv   2,000 rows, committed
│   ├── sample_tickets.csv        2,000 rows, committed
│   └── *.csv                     the full 1.5M-row files (gitignored, 766MB / 758MB)
├── models/                    trained pickles (gitignored — rebuild in ~2.5 min)
├── reports/
│   ├── eda.json               every aggregate the business case cites
│   ├── verify.json            26/26
│   └── logs/                  raw training output
└── notebooks/
```

## Running it

Everything reads paths **relative to the repository root** — run from there.

```bash
pip install -r requirements.txt

python src/eda.py             # aggregates      -> reports/eda.json     (~90s)
python src/train_pipeline.py  # all 7 models    -> models/*.pkl         (~2.5 min)
python src/verify.py          # 26 checks       -> reports/verify.json  (~2 min)
python src/cost_model.py      # headcount + sensitivity
```

Scoring new tickets needs only intake-time columns — no label, no issue type, no complexity:

```python
from pipeline import TicketPipeline
pipe = TicketPipeline.load("models/ticket_pipeline.pkl")
pipe.process(new_tickets)   # -> lane, probability, department, guardrail flag
```

---

## The pipeline

Three stages. Each sees only what would exist at that moment in a live system.

```
ticket text + CRM record
        │
   [1] UNDERSTANDING     intent · category · complexity · priority
        │                (+ confidence, so stage 2 can distrust its own upstream)
        ▼
   [2] DECISION          P(ai solvable) · P(sla breach)
        │                fed stage-1 PREDICTIONS, never the true columns
        ▼
   [0] GUARDRAIL         14 issue types an AI may never close — overrides everything
        │
        ├── auto-solve            33.3%   precision 0.918
        ├── AI draft + approve    40.9%
        ├── guardrail → human     13.7%
        └── human                 12.2%
                │
           [3] ROUTER    which of 5 teams
```

| Stage | Model | Output | Result | Baseline |
|---|---|---|---|---|
| 1 | Intent | 100 classes | acc 1.000 | 0.017 |
| 1 | Category | 11 classes | acc 1.000 | 0.097 |
| 1 | Priority | 4 classes | acc 0.470 | 0.336 |
| 1 | Complexity | 10 levels | MAE 1.112 | 2.094 |
| 2 | **Solvability gate** | binary | **AUC 0.880** | 0.881 *(lookup)* |
| 2 | **SLA-breach risk** | binary | **AUC 0.922** | 10.4% base rate |
| 3 | Department router | 5 teams | acc 0.809 | 0.363 |

**Split discipline.** Stage 1 fits on TRAIN-A. Stage 1 then predicts on TRAIN-B, and stage 2
fits on *those predictions*. TEST is held out from both. Without this, stage 2 would learn to
trust an intent signal that looks perfect on stage 1's own training rows and is far noisier in
production.

**The number this architecture exists to produce.** A gate handed the true complexity score
reaches AUC 0.891. The same gate fed its own inferred complexity — carrying ±1.1 levels of
error — reaches 0.880. That **−0.011** is the cost of the cascade, and it is what separates
"features that sit in a spreadsheet" from "features that exist at decision time."

---

## Two results reported because they are inconvenient

**1. The model barely beats a lookup table.**

| Method | Accuracy |
|---|---:|
| Predict everything human | 52.3% |
| Hand-written intuition | 66.5% |
| Rule: complexity ≤ 4 | 75.1% |
| Rule: issue type ≥ 50% historical | **80.1%** |
| LightGBM over text + 20 features | **80.4%** |

Six hundred trees and a TF-IDF vocabulary buy **three-tenths of a point** over a rule you could
write on an index card. That is the honest measure of what ML adds here, and the reason the rest
of the project is about economics.

**2. Automation alone underdelivers.**

Auto-solving every ticket the model is confident about saves **5.4 agents** — not the 12 the
matched dataset says are available. A perfect classifier would save 14.2. The gap is destroyed
by misroutes: at 92% precision, the 8% of failures cost 13.9 hours each, and that eats the gain.
**No threshold fixes this** — every setting from 0.45 to 0.90 lands within half an agent.

The money is in the lane where a human still checks the work:

| Strategy | Agents | Saved | Net / year |
|---|---:|---:|---:|
| No AI (baseline) | 44.0 | — | — |
| Auto-solve only | 38.6 | 5.4 | $146,637 |
| + draft tier, pessimistic | 38.1 | 5.9 | $141,383 |
| **+ draft tier, central** | **34.0** | **10.0** | **$264,425** |
| + draft tier, optimistic | 29.9 | 14.1 | $387,467 |

*At $30,000 fully loaded per agent-year and $0.10 per AI attempt — both inputs, not findings.
AI inference is ~11% of the labour saved; compute is not the cost driver.*

---

## Verification

`python src/verify.py` — **26 checks, 0 failures**, recomputed from source, non-zero exit on any
failure so it works as a CI gate. It covers data integrity, leakage, guardrail coverage,
calibration, and a complexity ablation that would expose the label leaking through a feature.

Leakage is structural rather than aspirational: features are an explicit **whitelist**, and the
feature builder raises if anything from the `LEAKY` set reaches the matrix.

```python
leaked = LEAKY & set(X.columns)
if leaked:
    raise RuntimeError(f"leakage: {sorted(leaked)} reached the features")
```

One check deserves a caveat rather than a victory lap. *"Guardrail fires identically on predicted
vs true issue type"* passes at 1.000000 — but only because intent classification is perfect on
templated text. On real tickets at 70–85% intent accuracy, roughly one in five security tickets
would be misclassified and slip past a guardrail keyed on a *predicted* type. The check is right
to exist because it would catch exactly that; here it confirms a property of the data, not the
safety of the design.

---

## What this project does not claim

- **Not evidence about real support desks.** It is a simulation study; see *About the data*.
- **No response generation.** The dataset contains no reply text, so any generator would be
  templates or a prompted LLM — neither trained on this data. Out of scope by choice.
- **The draft tier's value is an assumption.** It is the one number that is not measured,
  because no such tier has ever run. The range ($141k–$387k) is wide enough to change the
  decision, which is why the [pilot protocol](https://claude.ai/code/artifact/693ea9cd-385c-455d-bc4d-4abcc9ac53e2)
  exists: 600 tickets, within-agent crossover, decision rule fixed before the first ticket routes.
  200 tickets — the intuitive size — returns an interval too wide to act on.

---

## How this project developed

1. **Compared the two files** and found they were not two datasets but one matched pair —
   identical in every column except routing. That reframed the whole problem: the counterfactual
   was already measured.
2. **Built a single classifier** for `auto_resolved`. AUC 0.891, accuracy 0.805. Then benchmarked
   it against a one-line lookup and found it won by 0.4pp — so accuracy was clearly not where the
   value was.
3. **Measured what a mistake costs.** AI-solved 1.71h, human 12.34h, failed-AI 30.46h. Controlled
   for complexity to rule out selection bias; the penalty held at all ten levels. This became the
   spine of the project.
4. **Built the cost model**, converting accuracy into headcount using measured effort curves
   rather than an assumed saving per ticket. Discovered automation alone captures only 5.4 of the
   14.2 available agents, and that the draft tier is where the value actually is.
5. **Designed the pilot** to measure the one remaining unknown — and found that the intuitive
   200-ticket size is underpowered, computing what it would actually take.
6. **Rebuilt as a three-stage pipeline** so that stage 2 consumes stage 1's predictions rather
   than ground-truth columns, and measured the cascade cost at −0.011 AUC.
7. **Added a guardrail and a verification suite**, adapted from a reference implementation of the
   same problem. The guardrail turned out to be nearly free: it blocks 36,820 tickets but the gate
   would only have automated 21 of them, because the model had already learned to decline
   money disputes and account compromise.

---

## Credits

Dataset designed and generated by **[@talgatkozahmetov](https://github.com/talgatkozahmetov)**.
The guardrail issue-type list and the verification-suite discipline are adapted from a parallel
implementation of this problem by **Marat Akylov**.
