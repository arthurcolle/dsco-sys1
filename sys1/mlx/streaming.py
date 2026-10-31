"""Hash-verified HTTP JSONL streams; RAM buffers only, no dataset disk cache.

Use HTTPS, or a loopback URL carried by an authenticated DSCO Fleet SSH tunnel.
Source identities and evaluation weights are immutable. A tiny local policy file
can change training probabilities atomically between optimizer batches.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import urllib.parse
import urllib.request
import zlib

import mlx.core as mx
import numpy as np
from sys1.mlx.data import Mixture,_tuples
from sys1.model.corpus_manifest import file_identity

CHUNK_BYTES=256*1024
DOCUMENT_BYTES=1024*1024
MANIFEST_BYTES=2*1024*1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('stream redirects are disabled')


def fetch(url,limit):
    parsed=urllib.parse.urlsplit(url)
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError('credentials/fragments forbidden in stream URLs')
    if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ['127.0.0.1','localhost','::1']):
        raise ValueError('stream requires HTTPS or an authenticated loopback tunnel')
    request=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
    with urllib.request.build_opener(NoRedirect).open(request,timeout=15) as response:
        data=response.read(limit+1)
    if len(data)>limit:raise ValueError('stream response exceeds byte cap')
    return data


def weights(values,names,allow_zero=False):
    if not isinstance(values,dict) or set(values)!=set(names) or any(type(v) not in [int,float] or not math.isfinite(v) or v<0 or (v==0 and not allow_zero) for v in values.values()):
        raise ValueError('finite valid weight required for every declared source')
    total=math.fsum(values.values())
    if not math.isfinite(total) or total<=0:raise ValueError('invalid weight total')
    result={k:v/total for k,v in values.items()}
    if any(result[k]==0 and values[k]>0 for k in values):raise ValueError('weight normalization underflow')
    return result


class Reader:
    def __init__(self,base,records,repeat=True):
        self.base,self.records,self.repeat=base,records,repeat
        self.fi,self.chunk,self.offset,self.epoch=0,0,0,0
        self._cache=b'';self._cached_key=None

    def state(self):return {'file':self.fi,'chunk':self.chunk,'offset':self.offset,'epoch':self.epoch}

    def load_state(self,s):
        if any(type(s.get(k)) is not int or s[k]<0 for k in ['file','chunk','offset','epoch']):raise ValueError('invalid stream cursor')
        if s['file']>=len(self.records):raise ValueError('stream file cursor exceeds inventory')
        rec=self.records[s['file']]
        if s['chunk']>len(rec['chunks']) or s['offset']>CHUNK_BYTES or (s['chunk']==len(rec['chunks']) and s['offset']):
            raise ValueError('stream chunk cursor exceeds inventory')
        if s['chunk']<len(rec['chunks']):
            size=min(CHUNK_BYTES,rec['bytes']-s['chunk']*CHUNK_BYTES)
            if s['offset']>size:raise ValueError('stream byte cursor exceeds chunk')
        self.fi,self.chunk,self.offset,self.epoch=(s[k] for k in ['file','chunk','offset','epoch'])
        self._cache=b'';self._cached_key=None

    def _piece(self):
        while True:
            if not self.repeat and self.epoch:raise StopIteration
            rec=self.records[self.fi]
            if self.chunk>=len(rec['chunks']):
                self.fi+=1;self.chunk=self.offset=0
                if self.fi==len(self.records):self.fi=0;self.epoch+=1
                continue
            key=(self.fi,self.chunk)
            if key!=self._cached_key:
                raw=fetch(self.base+'/'+rec['resource']+'/'+str(self.chunk),CHUNK_BYTES)
                expected=min(CHUNK_BYTES,rec['bytes']-self.chunk*CHUNK_BYTES)
                if len(raw)!=expected or hashlib.sha256(raw).hexdigest()!=rec['chunks'][self.chunk]:
                    raise ValueError('stream chunk identity differs; no fallback data used')
                self._cache,self._cached_key=raw,key
            if self.offset==len(self._cache):self.chunk+=1;self.offset=0;continue
            end=self._cache.find(b'\n',self.offset)
            end=len(self._cache) if end<0 else end+1
            result=self._cache[self.offset:end];self.offset=end
            return result

    def next_doc(self):
        fragments=[];size=0;initial_epoch=self.epoch
        while True:
            try:piece=self._piece()
            except StopIteration:
                if fragments:break
                raise
            if self.epoch!=initial_epoch and not fragments:
                initial_epoch=self.epoch
            fragments.append(piece);size+=len(piece)
            if size>DOCUMENT_BYTES:raise ValueError('stream document exceeds 1 MiB')
            if piece.endswith(b'\n'):break
        row=json.loads(b''.join(fragments))
        if not isinstance(row,dict) or not isinstance(row.get('text'),str) or not row['text']:
            raise ValueError('stream JSONL requires nonempty text')
        return row['text']


class Packed:
    def __init__(self,reader,tokenizer,eos,window):
        self.reader,self.tokenizer,self.eos,self.window=reader,tokenizer,eos,window
        self.buffer=[]
    def next_window(self):
        while len(self.buffer)<self.window:
            ids=self.tokenizer.encode(self.reader.next_doc(),add_special_tokens=False).ids
            if len(ids)>262144:raise ValueError('stream document exceeds token RAM cap')
            self.buffer.extend(ids);self.buffer.append(self.eos)
        result,self.buffer=self.buffer[:self.window],self.buffer[self.window:]
        return result
    def state(self):return {'reader':self.reader.state(),'buffer':self.buffer}
    def load_state(self,state):
        buf=state.get('buffer')
        if not isinstance(buf,list) or len(buf)>262144+self.window or any(type(v) is not int or not 0<=v<self.tokenizer.get_vocab_size() for v in buf):
            raise ValueError('invalid stream token buffer')
        self.reader.load_state(state['reader']);self.buffer=list(buf)


class DynamicMixture(Mixture):
    def __init__(self,*args,policy_path=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.policy_path=Path(policy_path) if policy_path else None
        self.policy_version=0;self.policy_sha256=None

    def refresh(self):
        if not self.policy_path:return
        with self.policy_path.open('rb') as f:raw=f.read(65537)
        if len(raw)>65536:raise ValueError('dynamic policy exceeds 64 KiB')
        sha=hashlib.sha256(raw).hexdigest()
        if sha==self.policy_sha256:return
        p=json.loads(raw);version=p.get('version')
        if p.get('schema')!='sys1-stream-policy-v1' or p.get('stream_manifest_sha256')!=self.data_id:
            raise ValueError('dynamic policy source identity differs')
        if type(version) is not int or version<=self.policy_version:
            raise ValueError('dynamic policy version must increase; rollback rejected')
        revised=weights(p.get('weights'),self.names,allow_zero=True)
        self.weights=revised;self.policy_version=version;self.policy_sha256=sha

    def batch(self,size):
        self.refresh()  # Exactly once, before constructing an optimizer batch.
        return super().batch(size)

    def state(self):
        return {**super().state(),'policy_version':self.policy_version,'policy_sha256':self.policy_sha256}

    def load_state(self,state):
        if state.get('data_id')!=self.data_id or set(state.get('sources',{}))!=set(self.names):raise ValueError('saved stream identity differs')
        restored=weights(state.get('weights'),self.names,allow_zero=True)
        if not math.isclose(math.fsum(state['weights'].values()),1.,abs_tol=1e-12):raise ValueError('saved weights must sum to one')
        drawn=state.get('drawn',{})
        if set(drawn)!=set(self.names) or any(type(v) is not int or v<0 for v in drawn.values()):raise ValueError('invalid stream sampler counts')
        self.rng.setstate(_tuples(state['rng']));self.weights=restored;self.drawn=dict(drawn)
        for name,source in self.sources.items():source.load_state(state['sources'][name])
        version=state.get('policy_version',0);sha=state.get('policy_sha256')
        if type(version) is not int or version<0:raise ValueError('invalid saved policy version')
        if version and (not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha)):
            raise ValueError('invalid saved policy digest')
        self.policy_version,self.policy_sha256=version,sha


def build_stream(url,expected_sha,tokenizer_path,tokenizer,seq,eval_seqs,seed,policy_path=None):
    raw=fetch(url,MANIFEST_BYTES);actual=hashlib.sha256(raw).hexdigest()
    if actual!=expected_sha:raise ValueError('stream manifest identity differs')
    manifest=json.loads(raw)
    if manifest.get('schema')!='sys1-http-corpus-v1' or manifest.get('chunk_bytes')!=CHUNK_BYTES:
        raise ValueError('unsupported stream manifest')
    if manifest.get('tokenizer_sha256')!=file_identity(tokenizer_path)['sha256']:raise ValueError('stream tokenizer differs')
    entries=manifest.get('sources')
    if not isinstance(entries,dict) or not 1<=len(entries)<=32:raise ValueError('invalid stream source count')
    base=url.rsplit('/',1)[0];eos=tokenizer.token_to_id('[EOS]')
    if eos is None:raise ValueError('tokenizer requires EOS')
    w=weights(manifest.get('objective_weights'),entries);sources={};held={};origin={}
    identity=hashlib.sha256((actual+file_identity(tokenizer_path)['sha256']).encode())
    for name,entry in entries.items():
        for split in ['train','validation']:
            records=entry.get(split)
            if not isinstance(records,list) or not 1<=len(records)<=1024:raise ValueError('invalid stream file count')
            for rec in records:
                resource=rec.get('resource','')
                if not isinstance(resource,str) or any(v in resource for v in ['..','?','#',':','\\']) or not resource.startswith('chunk/'):
                    raise ValueError('unsafe stream resource')
                if rec.get('ends_newline') is not True:raise ValueError('stream files require trailing newline')
                if type(rec.get('bytes')) is not int or rec['bytes']<=0 or len(rec.get('chunks',[]))!=math.ceil(rec['bytes']/CHUNK_BYTES):raise ValueError('invalid chunk inventory')
                if any(not isinstance(s,str) or len(s)!=64 or any(c not in '0123456789abcdef' for c in s) for s in rec['chunks']):raise ValueError('invalid chunk hash')
        sources[name]=Packed(Reader(base,entry['train']),tokenizer,eos,seq+1)
        reader=Packed(Reader(base,entry['validation'],repeat=False),tokenizer,eos,seq+1)
        rng=random.Random(seed+zlib.crc32(name.encode()));selected=[];indices=[];seen=0
        while True:
            try:value=reader.next_window()
            except StopIteration:break
            slot=seen if seen<eval_seqs else rng.randrange(seen+1)
            if slot<eval_seqs:
                if slot==len(selected):selected.append(value);indices.append(seen)
                else:selected[slot],indices[slot]=value,seen
            seen+=1
        if not selected:raise ValueError('no complete validation windows for '+name)
        ordered=sorted(zip(indices,selected));values=np.asarray([v for _,v in ordered],dtype=np.int32)
        held[name]=mx.array(values);identity.update(name.encode());identity.update(values.tobytes())
        origin[name]={'stream_manifest_sha256':actual,'validation_sampling':{'method':'full-pass reservoir','complete_pass':True,
                      'total_windows':seen,'sampled_windows':len(selected),'selected_indices':[i for i,_ in ordered]}}
    mix=DynamicMixture(sources,w,seed,actual,policy_path=policy_path)
    return mix,held,origin,identity.hexdigest()
