---
license: cc-by-4.0
language:
  - en
size_categories:
  - 1M<n<10M
task_categories:
  - tabular-classification
  - text-classification
task_ids:
  - multi-class-classification
  - intent-classification
tags:
  - synthetic
  - customer-support
  - counterfactual
  - causal-inference
  - cost-sensitive-learning
  - operations-research
  - tabular
pretty_name: Support Tickets with an AI-Automation Counterfactual
configs:
  - config_name: with_ai
    data_files: with_ai/*.parquet
  - config_name: humans_only
    data_files: humans_only/*.parquet
---

# Support Tickets with an AI-Automation Counterfactual

**1.5M synthetic support tickets in two matched worlds** — one where an AI assistant handles part
of the queue, one staffed entirely by humans. Same tickets, same customers, same attributes.
Only the routing differs.

Most support datasets give you one world and leave you guessing about the other. This one gives
you both, so questions like *"what would this have cost without automation?"* are **measured
rather than estimated**.

```python
from datasets import load_dataset

ai    = load_dataset("s2pidape/support-ticket-dataset", "with_ai")
human = load_dataset("s2pidape/support-ticket-dataset", "humans_only")
```

## ⚠️ This data is synthetic

I generated it with a simulator I wrote. **Nothing here is evidence about real support desks.**
It is a sandbox for methods, and the honest limitations are listed near the bottom — please read
them before deciding whether it fits your problem.

## The two configs

| Config | Rows | Cols | What it is |
|---|---:|---:|---|
| `with_ai` | 1,500,000 | 39 | Routed with an AI assistant. Carries the label `auto_resolved`. Staffed by **32 human agents** + AI. |
| `humans_only` | 1,500,000 | 38 | The same tickets, same order, every attribute identical — routed to people only. Staffed by **44 agents**. |

The two differ in exactly **three columns**: `assigned_team`, `assigned_to`, `resolved_by`.
Everything else — timings, SLA outcomes, CSAT, descriptions — is byte-identical, and `ticket_id`
aligns row-for-row. `with_ai` additionally carries `auto_resolved` and `escalated_to_human`;
`humans_only` carries `escalated` (identical in content to `escalated_to_human`).

**44 → 32 agents is the counterfactual the pair encodes.**

## What you can do with it

- **Cost-sensitive learning.** The main draw. Most classification datasets assume symmetric
  error costs. Here they are wildly asymmetric and *measurable*: a ticket the AI attempts and
  drops takes **30.5h** to resolve versus **12.3h** if a human had taken it from the start —
  so a false positive costs more than a true positive saves. Datasets that let you practice
  this are scarce.
- **Counterfactual / causal inference.** Compute the effect of automation directly from the
  matched pair instead of estimating it.
- **A teaching set for "ML is not always the answer."** A one-line rule (*automate this issue
  type if it historically self-resolved ≥50%*) scores **80.1%** accuracy. A tuned LightGBM over
  text plus 20 features scores **80.4%**. The label is stochastic, so ~80% is a genuine ceiling.
  Rare to be able to demonstrate that with real numbers.
- **Ordinary supervised tasks.** Binary classification (`auto_resolved`), 100-class intent
  (`issue_type_id`), 5-class routing (`assigned_team`), regression on `resolution_time_hours`,
  SLA-breach prediction, queueing/survival analysis.

## Schema

`ticket_id` · `created_at` · `issue_description` · `category` · `issue_type_id` · `product` ·
`channel` · `region` · `language` · `customer_name` · `customer_email` · `customer_age` ·
`customer_gender` · `subscription_type` · `customer_tenure_months` · `previous_tickets` ·
`customer_segment` · `priority` · `issue_complexity_score` · `assigned_team` · `assigned_to` ·
`status` · `auto_resolved`\* · `escalated_to_human`\* / `escalated`\*\* · `escalation_reason` ·
`reopen_count` · `first_response_time_hours` · `resolution_time_hours` · `paused_hours` ·
`resolution_wallclock_hours` · `first_resolved_at` · `resolved_at` · `closed_at` · `resolved_by` ·
`resolution_notes` · `sla_target_hours` · `sla_breached` · `sla_breach_margin` ·
`customer_satisfaction_score`

<sub>\* `with_ai` only  ·  \*\* `humans_only` only</sub>

**Names and emails are generated, not real people.** No PII.

### Leakage warning

If you model `auto_resolved`, these columns **are** the answer and must be dropped:
`escalated_to_human`, `escalation_reason`, `assigned_team`, `assigned_to`, `resolved_by`,
`resolution_notes`, `status`, and every post-resolution timing/SLA/CSAT field. Train only on what
exists at intake.

**Also drop the 152,745 rows where `auto_resolved` is null** — those tickets are still open. Their
outcome is *unknown*, not negative; folding them into the negative class biases the model.
That leaves **1,347,255** labelled rows, 47.7% positive.

## Signal in the data

| Field | Behaviour |
|---|---|
| `issue_complexity_score` | Strongest signal. Solve rate falls monotonically 91.6% (level 1) → 5.5% (level 10). |
| `issue_type_id` | 100 types spanning 0.9% → 93.9% solve rate. **No type is pure** — none is 0% or 100%. |
| `category` | 24.2% (Security) → 70.7% (Onboarding). |
| `priority` | 23.9% (Urgent) → 69.1% (Low). |
| `product`, `channel`, `customer_segment`, `subscription_type` | **Pure noise** — all within 0.3pp of the 47.7% mean. Included deliberately as distractors. |

## Known limitations

Stated plainly so you can judge fit before downloading 176 MB:

1. **`issue_description` is template-generated and one-to-one with `issue_type_id`.** Intent
   classification scores a meaningless **100%**; only ~2,500 distinct TF-IDF features exist
   across 1.4M documents. **Do not use this for NLP benchmarking** — real intent models run
   70–85%.
2. **`resolution_notes` has only 10 distinct values** across 1.4M rows, and they are internal
   summaries rather than customer replies. There is **no reply text**, so nothing generative can
   be trained here.
3. **Perfectly stationary.** Solve rate sits between 0.475 and 0.480 across all twelve quarters.
   Train on 2022–23, test on 2024, and AUC moves by 0.0003. Useless for drift/concept-shift work.
4. **Four columns carry no signal at all** (see table above). Intentional, but know it going in.
5. **The counterfactual file's timing columns are copied, not re-simulated.** `humans_only` is
   valid for the **headcount** counterfactual (44 vs 32 agents) but its `resolution_time_hours`
   are the AI-world values. Do not read it as "how long humans would have taken."

## Stats

- **Span:** 2022-01-01 → 2024-12-30 (3.0 years), volume growing 28,122 → 54,975 per month
- **Automation rate** 47.7% · **escalation rate** 30.5% · **SLA breach** 10.4% · **reopen** 11.4%
- **Mean CSAT** 3.63/5 · **median first response** 0.45h · **median resolution** 1.82h
- 11 categories · 100 issue types · 6 products · 5 channels · 6 languages · 5 teams

## Reference implementation

A full analysis pipeline built on this dataset — three-stage triage, guardrail, cost model,
26-check verification suite — is at
**[github.com/s2pidape/ai-solvable](https://github.com/s2pidape/ai-solvable)**, with a written
[business case](https://claude.ai/code/artifact/270c52b9-d758-496e-aad3-640cf9b419eb).

## Citation

```bibtex
@misc{support_tickets_ai_counterfactual,
  title  = {Support Tickets with an AI-Automation Counterfactual},
  author = {Talgat Kozahmetov},
  year   = {2026},
  url    = {https://huggingface.co/datasets/s2pidape/support-ticket-dataset}
}
```

## License

**CC BY 4.0** — free to use, share, and adapt, including commercially, with attribution.
