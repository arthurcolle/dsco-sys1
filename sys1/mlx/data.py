"""Torch-free, resumable token streams for explicit verified corpus manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import zlib

import mlx.core as mx
import numpy as np

from sys1.model.corpus import DocReader, ENCODE_BATCH, MAX_DOC_CHARS
from sys1.model.corpus_manifest import file_identity, load_manifest


def _tuples(value):
    return tuple(map(_tuples,value)) if isinstance(value,list) else value


class PackedSource:
    def __init__(self, files, tokenizer, eos, window):
        self.reader = DocReader(files,"<|endoftext|>")
        self.tokenizer, self.eos, self.window = tokenizer,eos,window
        self.buffer = []

    def next_window(self):
        while len(self.buffer) < self.window:
            docs = [self.reader.next_doc() for _ in range(ENCODE_BATCH)]
            for encoded in self.tokenizer.encode_batch(docs,add_special_tokens=False):
                self.buffer.extend(encoded.ids)
                self.buffer.append(self.eos)
        result, self.buffer = self.buffer[:self.window],self.buffer[self.window:]
        return result

    def state(self):
        return {"reader":self.reader.state(),"buffer":list(self.buffer)}

    def load_state(self, state):
        if not isinstance(state.get("buffer"),list) or any(type(x) is not int or not 0 <= x < self.tokenizer.get_vocab_size() for x in state["buffer"]):
            raise ValueError("invalid saved token buffer")
        self.reader.load_state(state["reader"])
        self.buffer = list(state["buffer"])


class Mixture:
    def __init__(self, sources, weights, seed, data_id):
        self.sources, self.names, self.data_id = sources,list(sources),data_id
        total = sum(weights.values())
        self.weights = {name:weights[name]/total for name in self.names}
        self.rng = random.Random(seed)
        self.drawn = {name:0 for name in self.names}

    def batch(self, batch_size):
        windows, labels = [],[]
        for _ in range(batch_size):
            index = self.rng.choices(range(len(self.names)),[self.weights[name] for name in self.names])[0]
            name = self.names[index]
            self.drawn[name] += 1
            windows.append(self.sources[name].next_window())
            labels.append(index)
        return mx.array(windows,dtype=mx.int32),mx.array(labels,dtype=mx.int32)

    def state(self):
        return {"data_id":self.data_id,"rng":self.rng.getstate(),"weights":dict(self.weights),"drawn":dict(self.drawn),
                "sources":{name:source.state() for name,source in self.sources.items()}}

    def load_state(self, state):
        if state.get("data_id") != self.data_id or set(state.get("sources",{})) != set(self.names):
            raise ValueError("saved corpus identity or sources differ")
        if set(state.get("weights",{})) != set(self.names) or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not np.isfinite(x) or x <= 0 for x in state["weights"].values()):
            raise ValueError("invalid saved sampler weights")
        if not np.isclose(sum(state["weights"].values()),1,rtol=0,atol=1e-12):
            raise ValueError("saved sampler weights do not sum to one")
        if set(state.get("drawn",{})) != set(self.names) or any(type(x) is not int or x < 0 for x in state["drawn"].values()):
            raise ValueError("invalid saved sampler counts")
        self.rng.setstate(_tuples(state["rng"]))
        self.weights, self.drawn = dict(state["weights"]),dict(state["drawn"])
        for name, source in self.sources.items():
            source.load_state(state["sources"][name])


def reservoir(paths, tokenizer, eos, window, count, seed):
    reader = DocReader(paths,"<|endoftext|>",repeat=False)
    rng, selected, indices, buffer, batch = random.Random(seed),[],[],[],[]
    seen, documents, characters = 0,0,0
    def consume(docs):
        nonlocal seen,buffer
        for encoded in tokenizer.encode_batch(docs,add_special_tokens=False):
            buffer.extend(encoded.ids);buffer.append(eos)
            cursor = 0
            while cursor+window <= len(buffer):
                index = seen;seen += 1
                slot = index if index < count else rng.randrange(seen)
                if slot < count:
                    value = buffer[cursor:cursor+window]
                    if len(selected) < count:
                        selected.append(value);indices.append(index)
                    else:
                        selected[slot],indices[slot] = value,index
                cursor += window
            buffer = buffer[cursor:]
    try:
        while True:
            try:doc = reader.next_doc()
            except StopIteration:break
            if len(doc) > MAX_DOC_CHARS:
                raise ValueError("validation document exceeds 1 MiB")
            batch.append(doc);documents += 1;characters += len(doc)
            if len(batch) >= ENCODE_BATCH or characters >= 8*1024**2:
                consume(batch);batch,characters = [],0
        if batch:consume(batch)
    finally:
        reader.close()
    ordered = sorted(zip(indices,selected))
    values = np.asarray([row for _,row in ordered],dtype=np.int32)
    if len(values) == 0:
        raise ValueError("source has no complete validation windows")
    metadata = {"method":"reservoir","seed":seed,"documents":documents,"total_windows":seen,
                "sampled_windows":len(values),"selected_window_indices":[i for i,_ in ordered],
                "complete_pass":True,"tail_tokens":len(buffer)}
    return values,metadata


def build_data(manifest_path, tokenizer_path, tokenizer, seq, eval_seqs, seed, weights=None):
    if seq < 1 or eval_seqs < 1:
        raise ValueError("sequence and validation sizes must be positive")
    plan,data_id,manifest = load_manifest(manifest_path)
    weights = weights or manifest.get("objective_weights")
    if not isinstance(weights,dict) or set(weights) != set(plan) or any(not isinstance(w,(int,float)) or isinstance(w,bool) or not np.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("positive weights are required for every manifest source")
    eos = tokenizer.token_to_id("[EOS]")
    if eos is None:
        raise ValueError("tokenizer must contain [EOS]")
    sources, heldout, origin = {},{},{}
    digest = hashlib.sha256(file_identity(Path(tokenizer_path))["sha256"].encode())
    for name,entry in plan.items():
        sources[name] = PackedSource(entry["train"],tokenizer,eos,seq+1)
        values,metadata = reservoir(entry["validation"],tokenizer,eos,seq+1,eval_seqs,seed+zlib.crc32(name.encode()))
        digest.update(json.dumps([name,list(values.shape)]).encode());digest.update(values.tobytes())
        heldout[name] = mx.array(values,dtype=mx.int32)
        origin[name] = {"corpus_manifest_sha256":data_id,"validation_sampling":metadata}
    return Mixture(sources,weights,seed,data_id),heldout,origin,digest.hexdigest()
