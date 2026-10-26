import copy

import pytest

from sys1.datasets.trajectory_statistics import summarize


def observations(call=.9,final=.9):
    rows=[]
    for seed in (1,2,3):
        for session in range(25):
            for role,ratio in [('assistant_call',call),('assistant_final',final)]:
                for arm in ('uniform','targeted'):
                    rows.append({'seed':seed,'session':str(session),'item_id':str(session)+role,
                                 'target_role':role,'task_stratum':'inspection','context_bucket':'long','arm':arm,
                                 'target_nll':(2+session/25)*(ratio if arm=='targeted' else 1)})
    return rows


def test_paired_crossed_bootstrap_and_practical_threshold():
    result=summarize(observations(),bootstrap=200)
    assert result['diagnostic_gate_passed'] and not result['production_quality_established']
    assert result['primary']['relative_gain']==pytest.approx(.1)
    assert result['primary']['gain_interval']==pytest.approx([.1,.1])
    assert result['primary']['sessions']==25 and result['primary']['seeds']==3
    assert not summarize(observations(1,1),bootstrap=200)['diagnostic_gate_passed']
    assert not summarize(observations(),bootstrap=200,coverage_complete=False)['diagnostic_gate_passed']


def test_good_aggregate_cannot_hide_answer_regression():
    result=summarize(observations(.7,1.1),bootstrap=200)
    assert result['primary']['relative_gain']==pytest.approx(.1)
    assert not result['slice_guards']['role:assistant_final']['passed']
    assert not result['diagnostic_gate_passed']


def test_long_session_does_not_become_many_independent_sessions():
    rows=[]
    for session,n,loss in [('a',100,1.),('b',1,9.)]:
        for i in range(n):
            for arm in ('uniform','targeted'):
                rows.append({'seed':1,'session':session,'item_id':session+str(i),'target_role':'assistant_call',
                             'task_stratum':'inspection','context_bucket':'short','arm':arm,'target_nll':loss})
    result=summarize(rows,bootstrap=200)
    assert result['primary']['uniform_nll']==5.
    assert result['primary']['sessions']==2
    assert not result['sufficient_clusters'] and not result['diagnostic_gate_passed']


@pytest.mark.parametrize('damage',['missing','duplicate','metadata','seed_items'])
def test_pairing_and_item_identity_fail_closed(damage):
    rows=observations()
    if damage=='missing':rows.pop()
    if damage=='duplicate':rows.append(copy.deepcopy(rows[0]))
    if damage=='metadata':rows[1]['session']='drift'
    if damage=='seed_items':rows=rows[2:]
    with pytest.raises(ValueError):summarize(rows,bootstrap=200)
