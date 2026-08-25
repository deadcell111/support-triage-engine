"""Full EDA pass -> eda.json. Everything the business-case page reports."""
import json, numpy as np, pandas as pd

C = ['ticket_id','created_at','category','issue_type_id','product','channel','region',
     'language','customer_segment','subscription_type','priority','issue_complexity_score',
     'auto_resolved','escalated_to_human','resolution_time_hours','first_response_time_hours',
     'customer_satisfaction_score','reopen_count','sla_breached','sla_target_hours','status']
d = pd.read_csv('data/ground_truth.csv', usecols=C, low_memory=False)
d['t'] = pd.to_datetime(d.created_at)
res = d[d.auto_resolved.notna()].copy()
res['grp'] = np.where(res.auto_resolved=='Yes','ai_ok',
             np.where(res.escalated_to_human=='Yes','ai_fail','human'))
O = {}

O['scale'] = dict(
    total=len(d), resolved=int(len(res)), open=int(d.auto_resolved.isna().sum()),
    span_start=str(d.t.min().date()), span_end=str(d.t.max().date()),
    years=round((d.t.max()-d.t.min()).days/365.25, 2),
    customers_est=int(d.ticket_id.nunique()),
    automation_rate=round((res.auto_resolved=='Yes').mean()*100, 1),
    escalation_rate=round((res.escalated_to_human=='Yes').mean()*100, 1),
    sla_breach_rate=round((res.sla_breached=='Yes').mean()*100, 1),
    mean_csat=round(res.customer_satisfaction_score.mean(), 2),
    median_resolution_h=round(res.resolution_time_hours.median(), 2),
    median_frt_h=round(res.first_response_time_hours.median(), 2),
    reopen_rate=round((res.reopen_count>0).mean()*100, 1),
)

# monthly volume + automation trend
m = d.groupby(d.t.dt.to_period('M'))
O['monthly'] = [{'month': str(k), 'tickets': int(v)} for k, v in m.size().items()]
mr = res.groupby(res.t.dt.to_period('M')).auto_resolved.apply(lambda s:(s=='Yes').mean()*100)
O['monthly_automation'] = [{'month': str(k), 'rate': round(v,2)} for k,v in mr.items()]

# distributions
for col, key in [('category','by_category'), ('channel','by_channel'),
                 ('region','by_region'), ('priority','by_priority'),
                 ('customer_segment','by_segment')]:
    g = res.groupby(col).agg(n=('ticket_id','size'),
                             auto=('auto_resolved', lambda s:(s=='Yes').mean()*100),
                             csat=('customer_satisfaction_score','mean'),
                             hours=('resolution_time_hours','mean'))
    O[key] = [{'name': str(k), 'n': int(r.n), 'auto': round(r.auto,1),
               'csat': round(r.csat,2), 'hours': round(r.hours,1)}
              for k, r in g.sort_values('n', ascending=False).iterrows()]

# the label gradient
g = res.groupby('issue_complexity_score').agg(
    n=('ticket_id','size'), auto=('auto_resolved', lambda s:(s=='Yes').mean()*100))
O['by_complexity'] = [{'level': int(k), 'n': int(r.n), 'auto': round(r.auto,1)}
                      for k, r in g.iterrows()]

it = res.groupby('issue_type_id').agg(
    n=('ticket_id','size'), auto=('auto_resolved', lambda s:(s=='Yes').mean()*100)
).sort_values('auto')
O['issue_type_extremes'] = dict(
    n_types=int(len(it)),
    lowest=[{'type':k,'auto':round(r.auto,1),'n':int(r.n)} for k,r in it.head(6).iterrows()],
    highest=[{'type':k,'auto':round(r.auto,1),'n':int(r.n)} for k,r in it.tail(6).iterrows()],
    n_pure=int(((it.auto==0)|(it.auto==100)).sum()))

# the three paths -- the core economic finding
p = res.groupby('grp').agg(
    n=('ticket_id','size'), hours=('resolution_time_hours','mean'),
    csat=('customer_satisfaction_score','mean'),
    sla=('sla_breached', lambda s:(s=='Yes').mean()*100),
    reopen=('reopen_count', lambda s:(s>0).mean()*100),
    frt=('first_response_time_hours','mean'))
O['paths'] = {k: {'n': int(r.n), 'hours': round(r.hours,2), 'csat': round(r.csat,2),
                  'sla': round(r.sla,1), 'reopen': round(r.reopen,1), 'frt': round(r.frt,2)}
              for k, r in p.iterrows()}

# penalty within complexity -- proves it is not selection bias
sub = res[res.grp!='ai_ok']
pv = sub.pivot_table(index='issue_complexity_score', columns='grp',
                     values='resolution_time_hours', aggfunc=['mean','size'])
pv.columns = ['fail_h','human_h','fail_n','human_n']
cs = sub.pivot_table(index='issue_complexity_score', columns='grp',
                     values='customer_satisfaction_score', aggfunc='mean')
O['penalty_by_complexity'] = [
    {'level': int(k), 'human_h': round(r.human_h,2), 'fail_h': round(r.fail_h,2),
     'penalty_h': round(r.fail_h-r.human_h,2), 'ratio': round(r.fail_h/r.human_h,2),
     'csat_human': round(cs.loc[k,'human'],2), 'csat_fail': round(cs.loc[k,'ai_fail'],2)}
    for k, r in pv.iterrows()]
w = pv.human_n + pv.fail_n
O['penalty_weighted_h'] = round(float(((pv.fail_h-pv.human_h)*w).sum()/w.sum()), 2)
O['penalty_weighted_csat'] = round(float(((cs.ai_fail-cs.human)*w).sum()/w.sum()), 3)

# the counterfactual pair
t2 = pd.read_csv('data/tickets.csv', usecols=['ticket_id','assigned_team','assigned_to'],
                 low_memory=False)
g1 = pd.read_csv('data/ground_truth.csv', usecols=['assigned_team','assigned_to'], low_memory=False)
O['counterfactual'] = dict(
    workers_without_ai=int(t2.assigned_to.nunique()),
    workers_with_ai=int(g1.assigned_to.nunique()),
    humans_with_ai=int(g1[g1.assigned_to!='AI Assistant'].assigned_to.nunique()),
    teams_without_ai=int(t2.assigned_team.nunique()),
    ai_handled=int((g1.assigned_team=='AI Assistant').sum()))

json.dump(O, open('reports/eda.json','w'), indent=1)
print("wrote eda.json")
print(json.dumps(O['scale'], indent=1))
print("paths:", json.dumps(O['paths'], indent=1))
print("counterfactual:", json.dumps(O['counterfactual'], indent=1))
