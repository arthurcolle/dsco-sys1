"""Paired, crossed seed/session bootstrap for role-aware selection diagnostics."""
from __future__ import annotations

from collections import defaultdict
import math


def summarize(records, *, bootstrap=2000, seed=41873, minimum_sessions=20, minimum_seeds=3,
              minimum_gain=.01, maximum_regression=.02, coverage_complete=True):
    import numpy as np
    if not 200<=bootstrap<=10000:raise ValueError('bootstrap budget must be 200..10000')
    pairs=defaultdict(dict);metadata={}
    for row in records:
        if row['arm'] not in ('uniform','targeted') or not math.isfinite(row['target_nll']) or row['target_nll']<=0:
            raise ValueError('invalid paired observation')
        key=(row['seed'],row['item_id'])
        if row['arm'] in pairs[key]:raise ValueError('duplicate paired observation')
        identity=tuple(row[k] for k in ('session','target_role','task_stratum','context_bucket'))
        if row['item_id'] in metadata and metadata[row['item_id']]!=identity:raise ValueError('item identity drift')
        metadata[row['item_id']]=identity;pairs[key][row['arm']]=row['target_nll']
    if not pairs or any(set(arms)!= {'uniform','targeted'} for arms in pairs.values()):raise ValueError('missing paired observation')
    seeds=sorted({key[0] for key in pairs})
    item_sets=[{key[1] for key in pairs if key[0]==s} for s in seeds]
    if any(items!=item_sets[0] for items in item_sets):raise ValueError('validation items differ across seeds')
    def estimate(items, alpha):
        cells=defaultdict(list)
        for (s,item),arms in pairs.items():
            if item in items:
                session,role,_,_=metadata[item]
                for arm,value in arms.items():cells[s,session,arm,role].append(value)
        sessions=sorted({metadata[item][0] for item in items})
        means={}
        for arm in ('uniform','targeted'):
            matrix=[]
            for s in seeds:
                values=[]
                for session in sessions:
                    roles=[np.mean(v) for (ss,gg,aa,_),v in cells.items() if (ss,gg,aa)==(s,session,arm)]
                    if not roles:raise ValueError('missing seed/session cell')
                    values.append(float(np.mean(roles)))
                matrix.append(values)
            means[arm]=np.array(matrix)
        baseline,candidate=means['uniform'],means['targeted']
        rng=np.random.default_rng(seed)
        si=rng.integers(0,len(seeds),size=(bootstrap,len(seeds),1))
        gi=rng.integers(0,len(sessions),size=(bootstrap,1,len(sessions)))
        draws=1-candidate[si,gi].mean(axis=(1,2))/baseline[si,gi].mean(axis=(1,2))
        interval=np.quantile(draws,[alpha/2,1-alpha/2]).tolist()
        return {'sessions':len(sessions),'seeds':len(seeds),'windows':len(items),
                'uniform_nll':float(baseline.mean()),'targeted_nll':float(candidate.mean()),
                'relative_gain':float(1-candidate.mean()/baseline.mean()),'gain_interval':interval,
                'paired_seed_wins':int((candidate.mean(axis=1)<baseline.mean(axis=1)).sum())}
    all_items=set(metadata);primary=estimate(all_items,.05)
    slices={}
    for field,index in [('role',1),('task',2),('context',3)]:
        for value in sorted({m[index] for m in metadata.values()}):
            slices[field+':'+value]={item for item,m in metadata.items() if m[index]==value}
    guards={name:estimate(items,.05/max(1,len(slices))) for name,items in slices.items()}
    for value in guards.values():
        value['adequate_sessions']=value['sessions']>=5
        value['passed']=value['adequate_sessions'] and value['gain_interval'][0]>=-maximum_regression
    enough=primary['sessions']>=minimum_sessions and primary['seeds']>=minimum_seeds
    return {'primary':primary,'slice_guards':guards,'method':'paired crossed bootstrap over seeds and whole sessions; role macro then session macro',
            'bootstrap_replicates':bootstrap,'bootstrap_seed':seed,'multiple_comparisons':'Bonferroni over all observed role/task/context guards; one primary comparison',
            'criteria':{'minimum_sessions':minimum_sessions,'minimum_seeds':minimum_seeds,'minimum_relative_gain':minimum_gain,
                        'maximum_source_regression':maximum_regression,'primary_lower_interval_above_zero':True},
            'coverage_complete':coverage_complete,'sufficient_clusters':enough,'missing_observations':0,
            'diagnostic_gate_passed':bool(enough and coverage_complete and primary['relative_gain']>=minimum_gain and
                   primary['gain_interval'][0]>0 and primary['paired_seed_wins']==len(seeds) and all(v['passed'] for v in guards.values())),
            'production_quality_established':False}
