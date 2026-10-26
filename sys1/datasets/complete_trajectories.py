"""Bounded, lossless chronicle reconstruction for derived training datasets.

The legacy exporter remains unchanged so previous corpora are reproducible.
Missing, changed, oversized or ambiguous records fail closed; nothing is clipped.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

from sys1.datasets.export_episodes import _user_texts

CONTROL_MARKERS = ('[BOS]','[EOS]','[ACT]','<|user|>','<|assistant|>','<|tool|>',
                   '<|tool:','<|end|>','<|json|>','<|/json|>')

class Incomplete(ValueError):
    pass


def argument_object(text):
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def constant(value):raise ValueError('nonfinite JSON number')
    parsed = json.loads(text,object_pairs_hook=pairs,parse_constant=constant)
    if not isinstance(parsed,dict):raise ValueError('argument must be an object')
    return parsed


class CompleteReader:
    def __init__(self, chronicle, *, max_blob_bytes=65536, max_request_bytes=2*1024**2,
                 max_session_bytes=16*1024**2):
        self.root = Path(chronicle).resolve()
        self.max_blob_bytes = max_blob_bytes
        self.max_request_bytes = max_request_bytes
        self.max_session_bytes = max_session_bytes
        self.counts = Counter()
        self.bytes_read = 0
        self.session_bytes = 0

    def blob(self, sha, *, request=False):
        if self.session_bytes >= self.max_session_bytes:
            raise Incomplete('session_payload_budget')
        if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
            raise Incomplete('missing_blob_reference')
        path = self.root/'blobs'/'sha256'/sha[:2]/sha[2:4]/(sha+'.blob')
        limit = self.max_request_bytes if request else self.max_blob_bytes
        try:
            with path.open('rb') as stream:
                raw = stream.read(limit+1)
        except OSError as exc:
            raise Incomplete('missing_blob') from exc
        self.bytes_read += len(raw)
        self.session_bytes += len(raw)
        if self.session_bytes > self.max_session_bytes:
            raise Incomplete('session_payload_budget')
        if len(raw) > limit:
            raise Incomplete('oversize_blob')
        if hashlib.sha256(raw).hexdigest() != sha:
            raise Incomplete('blob_hash_mismatch')
        try:
            return raw.decode('utf-8', errors='strict')
        except UnicodeError as exc:
            raise Incomplete('invalid_blob_utf8') from exc

    def request_goal(self, sha):
        try:
            request = json.loads(self.blob(sha, request=True))
        except (ValueError, TypeError) as exc:
            if isinstance(exc, Incomplete):raise
            raise Incomplete('invalid_request_json') from exc
        if not isinstance(request, dict):raise Incomplete('invalid_request_json')
        texts = []
        _user_texts(request.get('messages') or request.get('input') or [], texts)
        for text in reversed(texts):
            text = text.strip()
            if text and not text.startswith(('Tool observation', '[Earlier tool output cleared', '<system')):
                if len(text.encode()) > self.max_blob_bytes:raise Incomplete('oversize_goal')
                return text
        raise Incomplete('missing_user_goal')

    def session(self, path, *, min_age_seconds=60):
        path = Path(path).resolve()
        if (self.root/'events') not in path.parents:raise ValueError('event path outside chronicle')
        before = path.stat()
        if before.st_size > 4*1024**2 or time.time()-before.st_mtime < min_age_seconds:
            self.counts['oversize_or_active_event_file'] += 1
            return [], None
        with path.open('rb') as stream:raw = stream.read(4*1024**2+1)
        if len(raw)>4*1024**2:
            self.counts['oversize_or_active_event_file'] += 1
            return [], None
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            self.counts['changing_event_file'] += 1
            return [], None
        self.bytes_read += len(raw)
        identity = {'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
        try:
            events = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
            if any(not isinstance(e,dict) or type(e.get('seq')) is not int for e in events):
                raise ValueError('event schema')
            seqs = [e['seq'] for e in events]
            if len(set(seqs)) != len(seqs):raise ValueError('duplicate sequence')
            events.sort(key=lambda e:e['seq'])
        except (UnicodeError, ValueError):
            self.counts['invalid_event_file'] += 1
            return [], identity
        self.counts['event_files_read'] += 1
        self.session_bytes = 0
        out = []
        current = None
        pending = {}
        used_spans = set()

        def close(reason):
            nonlocal current
            if current:
                current['end'] = reason
                if pending:current['rejection'] = current.get('rejection') or 'unresolved_tool_calls'
                out.append(current)
            current = None
            pending.clear()

        def start(event, goal):
            return {'session':event.get('session_id'), 'goal':goal, 'steps':[], 'final':None,
                    'rejection':None, 'end':None}

        for event in events:
            kind = event.get('event_type')
            payload = event.get('payload')
            if kind not in {'user.message','llm.request.created','llm.response.completed',
                            'tool.call.created','tool.call.completed'}:continue
            if not isinstance(payload,dict):
                if current:current['rejection'] = 'invalid_event_payload'
                continue
            if kind == 'user.message':
                close('next_user_message')
                current = start(event, '')
                try:current['goal'] = self.blob(payload.get('blob_sha256'))
                except Incomplete as exc:current['rejection'] = str(exc)
                continue
            if kind == 'llm.request.created' and current is None:
                current = start(event, '')
                try:current['goal'] = self.request_goal(payload.get('request_blob_sha256'))
                except Incomplete as exc:current['rejection'] = str(exc)
                continue
            if current is None:continue
            if event.get('session_id') != current['session']:
                current['rejection'] = 'mixed_session_events'
            try:
                if kind == 'tool.call.created':
                    span = event.get('span_id')
                    if not isinstance(span,str) or not span or span in used_spans:
                        raise Incomplete('ambiguous_tool_span')
                    used_spans.add(span)
                    args = self.blob(payload.get('args_blob_sha256'))
                    try:parsed = json.loads(args)
                    except ValueError as exc:raise Incomplete('invalid_tool_arguments') from exc
                    if not isinstance(parsed,dict):raise Incomplete('nonobject_tool_arguments')
                    try:argument_object(args)
                    except ValueError as exc:raise Incomplete('invalid_tool_arguments') from exc
                    tool = payload.get('tool_name')
                    if not isinstance(tool,str) or not tool:raise Incomplete('missing_tool_name')
                    step = {'tool':tool, 'args':args, 'result':None, 'ok':None}
                    current['steps'].append(step)
                    pending[span] = step
                elif kind == 'tool.call.completed':
                    step = pending.pop(event.get('span_id'), None)
                    if step is None or step['tool'] != payload.get('tool_name'):
                        raise Incomplete('unmatched_tool_result')
                    if type(payload.get('ok')) is not bool:raise Incomplete('unknown_tool_status')
                    step.update(result=self.blob(payload.get('result_blob_sha256')), ok=payload['ok'])
                elif kind == 'llm.response.completed' and payload.get('finish_reason') != 'tool_use':
                    current['final'] = {'finish_reason':payload.get('finish_reason'),
                                        'text':self.blob(payload.get('output_blob_sha256'))}
                    close('final_answer')
            except Incomplete as exc:
                if current:current['rejection'] = current.get('rejection') or str(exc)
        close('session_end')
        return out, identity


def rejection(ep):
    if ep.get('rejection'):return ep['rejection']
    if not isinstance(ep.get('session'),str) or not ep['session']:return 'missing_session'
    final = ep.get('final') or {}
    if ep.get('end') != 'final_answer' or final.get('finish_reason') not in {'end_turn','stop','eos_token'}:
        return 'incomplete_final'
    if not isinstance(ep.get('goal'),str) or len(ep['goal'].strip()) < 20:return 'short_goal'
    if not isinstance(final.get('text'),str) or len(final['text'].strip()) < 64:return 'short_final'
    steps = ep.get('steps') or []
    if not 1 <= len(steps) <= 128:return 'tool_count'
    if any(type(s.get('ok')) is not bool or not isinstance(s.get('result'),str) or
           not isinstance(s.get('args'),str) for s in steps):return 'incomplete_tool'
    if any(not isinstance(s.get('tool'),str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_:/.-]*',s['tool']) for s in steps):
        return 'invalid_tool_name'
    fields=[ep['goal'],final['text']]+[s[k] for s in steps for k in ('args','result')]
    if any(marker in field for field in fields for marker in CONTROL_MARKERS):return 'embedded_control_marker'
    if sum(s['ok'] for s in steps)/len(steps) < .8:return 'tool_failure_rate'
    return None
