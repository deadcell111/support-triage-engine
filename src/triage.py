"""Score a ticket description through the whole pipeline, from the terminal.

    python src/triage.py "I was charged twice this month"
    python src/triage.py                      # interactive loop

Only the description is required. Customer fields default to dataset medians,
since someone typing a ticket at a prompt does not have a CRM record to hand.
"""
import os
import sys

import pandas as pd

# Resolve everything against the repo root (the parent of src/), so the script
# runs from any working directory rather than only from the project root.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pipeline import GUARDRAIL_ISSUE_TYPES, TicketPipeline  # noqa: E402

MODEL = os.path.join(ROOT, "models", "ticket_pipeline.pkl")

# dataset medians -- neutral stand-ins for a real customer record
DEFAULTS = dict(created_at="2024-06-15 10:00:00", product="Web Portal",
                channel="Email", region="North America", language="English",
                customer_gender="Female", subscription_type="Basic",
                customer_segment="Individual", customer_age=42,
                customer_tenure_months=24, previous_tickets=2)

# mean human handling hours by complexity, measured in reports/eda.json
HUMAN_H = {1: 3.06, 2: 4.82, 3: 6.48, 4: 8.28, 5: 10.71,
           6: 12.96, 7: 16.45, 8: 19.74, 9: 25.14, 10: 30.96}
FAIL_H = {1: 12.95, 2: 14.79, 3: 16.12, 4: 19.02, 5: 22.18,
          6: 27.29, 7: 32.76, 8: 37.24, 9: 40.92, 10: 46.38}

BAR = lambda p, n=22: "█" * round(p * n) + "░" * (n - round(p * n))
LANE = {"auto_solve": ("AUTO-SOLVE", "AI handles it end to end"),
        "ai_draft": ("AI DRAFT", "AI writes it, a human approves before send"),
        "guardrail": ("GUARDRAIL", "blocked from AI regardless of confidence"),
        "human": ("HUMAN", "straight to a person")}


def triage(pipe, text, **over):
    row = {**DEFAULTS, **over, "ticket_id": 0, "issue_description": text}
    out = pipe.process(pd.DataFrame([row])).iloc[0]
    cx = int(out.issue_complexity_score)
    lane, why = LANE[out.lane]

    print(f"\n\033[1m TICKET \033[0m {text[:78]}")
    print(f"\n  \033[2m1 UNDERSTANDING\033[0m")
    print(f"    issue type    {out.issue_type_id}")
    print(f"    category      {out.category}")
    print(f"    priority      {out.priority}")
    print(f"    complexity    {cx}/10")
    print(f"    confidence    {BAR(out.intent_conf)}  {out.intent_conf:.0%}")

    print(f"\n  \033[2m2 DECISION\033[0m")
    print(f"    P(AI solves)  {BAR(out.p_ai_solvable)}  {out.p_ai_solvable:.1%}")
    print(f"    P(SLA breach) {BAR(out.p_sla_breach)}  {out.p_sla_breach:.1%}")

    if out.guardrail_blocked:
        print(f"\n  \033[1;31m  GUARDRAIL FIRED\033[0m  "
              f"'{out.issue_type_id}' is on the {len(GUARDRAIL_ISSUE_TYPES)}-type "
              f"never-automate list")

    colour = {"auto_solve": "32", "ai_draft": "36",
              "guardrail": "31", "human": "33"}[out.lane]
    print(f"\n  \033[1;{colour}m→ {lane}\033[0m  \033[2m{why}\033[0m")
    if out.department:
        print(f"    routed to     {out.department}")
    print(f"    if a human does it   ~{HUMAN_H[cx]:.1f}h")
    if out.lane in ("auto_solve", "ai_draft"):
        print(f"    if the AI fails      ~{FAIL_H[cx]:.1f}h  "
              f"\033[2m(+{FAIL_H[cx]-HUMAN_H[cx]:.1f}h penalty)\033[0m")
    print()


if __name__ == "__main__":
    if not os.path.exists(MODEL):
        sys.exit(f"No model at {MODEL}\n"
                 f"Build it first:  python {os.path.join('src','train_pipeline.py')}"
                 f"   (run from {ROOT})")
    pipe = TicketPipeline.load(MODEL)
    if len(sys.argv) > 1:
        triage(pipe, " ".join(sys.argv[1:]))
    else:
        print("Type a ticket description. Ctrl-C to quit.")
        try:
            while True:
                t = input("\n\033[1m>\033[0m ").strip()
                if t:
                    triage(pipe, t)
        except (KeyboardInterrupt, EOFError):
            print()
