"""Standalone, network-free HTML; validates identities before comparing runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import digest


def validate(record):
    if record.get("schema") != "sys1-neuron-result-v1" or record.get("status") != "complete":
        raise ValueError("only completed diagnostic records can be displayed")
    plan = record["plan"]
    if digest(plan) != record["fingerprint"] or digest(record["results"]) != record["results_digest"] or digest(plan["comparison"]) != plan["comparison_key"]:
        raise ValueError("report identity or result digest differs")
    return record


def render(records, path):
    records = [validate(r) for r in records]
    if not records or len({r["plan"]["comparison_key"] for r in records}) != 1:
        raise ValueError("comparison requires identical probes, tokenizer, architecture, precision, backend and options")
    # JSON is data, never executable HTML; token/probe strings are inserted using textContent only.
    payload = json.dumps(records, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = TEMPLATE.replace("__DATA__", payload)
    path = Path(path)
    if path.suffix.lower() != ".html":
        raise ValueError("report output must be a .html file")
    with path.open("w", encoding="utf-8") as stream:
        stream.write(html)
    path.chmod(0o600)


TEMPLATE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<title>SystemOne · neuron diagnostics</title><style>
:root{color-scheme:dark;font:15px system-ui;background:#101820;color:#dce8ee}body{margin:0 auto;max-width:1320px;padding:32px}
h1{font-size:30px;margin:8px 0}h2{font-size:17px}p{line-height:1.6}small,.muted{color:#9baebc}.tag{color:#66dcca;text-transform:uppercase;letter-spacing:.13em;font-size:12px}
.grid{display:grid;grid-template-columns:minmax(430px,1fr) minmax(350px,1fr);gap:22px}.box{background:#17242e;border:1px solid #2d414e;padding:20px;border-radius:12px;margin:16px 0}
select,button{font:inherit;background:#213541;color:inherit;border:1px solid #466271;border-radius:7px;padding:9px;margin:5px 12px 5px 0}button{cursor:pointer}.selected{border-color:#66dcca}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.65 ui-monospace,monospace;color:#d4e2e8}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 5px;text-align:right;border-bottom:1px solid #2d414e}th:first-child,td:first-child{text-align:left}tbody tr{cursor:pointer}tbody tr:hover{background:#223946}
.trace{font:13px/2.4 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.trace span{padding:3px 1px;border-radius:3px}.number{font-variant-numeric:tabular-nums}.positive{color:#f4be7a}.negative{color:#66dcca}#identity{overflow-wrap:anywhere}footer{padding:20px 0;font-size:12px;color:#9baebc}
@media(max-width:850px){body{padding:15px}.grid{display:block}table{font-size:12px}.box{padding:14px}}
</style><body><div class="tag">SystemOne / offline research</div><h1>Neuron diagnostics</h1>
<p class="muted">Natural activations, causal ablations and matched perturbations on a fixed continuation. These are exploratory measurements, with no training changes or semantic labels.</p>
<label>Checkpoint <select id="run"></select></label><label>Probe <select id="probe"></select></label>
<div class="grid"><section><div class="box"><div class="tag" id="stratum"></div><h2 id="probe-title"></h2><pre id="prompt"></pre><small>Scored continuation</small><pre id="target"></pre><p id="loss" class="number"></p><small id="pair"></small></div>
<div class="box"><h2>Neuron sensitivity</h2><small>ΔNLL = intervention − baseline. Positive means removal hurt this continuation. Click a row to inspect its trace.</small><table><thead><tr><th>Unit</th><th>Ablation Δ</th><th>Random Δ mean</th><th>Across checkpoints Δ</th></tr></thead><tbody id="neurons"></tbody></table></div></section>
<section><div class="box"><h2 id="unit-title">Activation trace</h2><p class="muted" id="selection"></p><div class="trace" id="trace"></div><p id="calibration" class="muted"></p><div id="doses"></div></div>
<div class="box"><h2>Identity & scope</h2><pre id="identity"></pre><p class="muted" id="limits"></p></div></section></div>
<footer>Standalone report · no model loading, agents, shell tools, external assets or network requests. A neuron with a large effect is a hypothesis to investigate, not a verified concept or circuit.</footer>
<script id="records" type="application/json">__DATA__</script><script>
const runs=JSON.parse(document.getElementById('records').textContent),$=id=>document.getElementById(id);
const put=(id,text)=>$(id).textContent=text,fmt=x=>x==null?'—':Number(x).toFixed(5);let ri=0,pi=0,selected=0;
for(const [i,r]of runs.entries()){const o=document.createElement('option');o.value=i;o.textContent='step '+(r.plan.model_instance.step??'?')+' · '+r.fingerprint.slice(0,10);$('run').append(o)}
for(const [i,r]of runs[0].results.entries()){const o=document.createElement('option');o.value=i;o.textContent=r.probe.id+' / '+r.probe.stratum;$('probe').append(o)}
function draw(){const r=runs[ri],s=r.results[pi],base=runs[0].results[pi];put('stratum',s.probe.stratum);put('probe-title',s.probe.id);put('prompt',s.probe.prompt);put('target',s.probe.continuation);put('loss','Continuation mean NLL: '+fmt(s.baseline.mean_nll)+' nats/token');
put('pair',runs.length>1?'Baseline NLL change from first checkpoint: '+fmt(s.baseline.mean_nll-base.baseline.mean_nll):'One checkpoint. Use the report CLI to compare compatible completed runs.');
put('identity',JSON.stringify({checkpoint:r.plan.checkpoint_files,step:r.plan.model_instance.step,backend:r.plan.comparison.backend,precision:r.plan.comparison.precision,fingerprint:r.fingerprint,role:r.plan.role},null,2));put('limits',r.plan.limitations.join(' '));
const sorted=[...s.neurons].sort((a,b)=>b.ablation_delta_nll-a.ablation_delta_nll);$('neurons').replaceChildren();
for(const [i,n]of sorted.entries()){const tr=document.createElement('tr'),prior=base.neurons.find(x=>x.layer===n.layer&&x.neuron===n.neuron),controls=n.random_direction_controls;const vals=['L'+n.layer+' · N'+n.neuron,fmt(n.ablation_delta_nll),controls.length?fmt(controls.reduce((a,x)=>a+x.delta_nll,0)/controls.length):'not measured',runs.length>1&&prior?fmt(n.ablation_delta_nll-prior.ablation_delta_nll):'—'];for(const v of vals){const td=document.createElement('td');td.textContent=v;tr.append(td)}tr.onclick=()=>unit(n);$('neurons').append(tr)}
if(sorted.length)unit(sorted[Math.min(selected,sorted.length-1)]);}
function unit(n){const s=runs[ri].results[pi];put('unit-title','L'+n.layer+' · N'+n.neuron+' / natural activation');put('selection',n.selection);$('trace').replaceChildren();const max=Math.max(...n.activation.map(Math.abs),1e-12);for(const [i,t]of s.tokens.entries()){const v=n.activation[i],span=document.createElement('span');span.textContent=t;span.title='position '+i+' · '+v.toFixed(6);span.style.backgroundColor=v>=0?'rgba(55,194,169,'+(.05+.65*Math.abs(v)/max)+')':'rgba(228,135,69,'+(.05+.65*Math.abs(v)/max)+')';$('trace').append(span)}put('calibration','Baseline residual median '+fmt(n.baseline_residual_median)+' · down-column norm '+fmt(n.down_column_norm));$('doses').replaceChildren();for(const a of n.additive){const p=document.createElement('p');p.textContent='Relative dose '+a.relative_dose+': ΔNLL '+fmt(a.delta_nll);$('doses').append(p)}}
$('run').onchange=e=>{ri=Number(e.target.value);draw()};$('probe').onchange=e=>{pi=Number(e.target.value);draw()};draw();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = [json.loads((root / "complete.json").read_text()) for root in args.runs]
    render(records, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
