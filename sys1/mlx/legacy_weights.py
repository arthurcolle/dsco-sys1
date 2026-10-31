"""Read tensor bytes from our legacy zip checkpoints without importing Torch.

Only a restricted metadata pickle is interpreted; model storages are streamed
unchanged into safetensors. Optimizer/RNG payloads are never executed or loaded.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import shutil
import struct
import zipfile

from sys1.model.corpus_manifest import file_identity


@dataclass
class Storage:
    kind: str
    key: str
    count: int


@dataclass
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple


class Ignored:
    def __init__(self, *args): pass
    def __setstate__(self, state): pass


def tensor(storage, offset, shape, stride, *rest):
    return Tensor(storage, offset, tuple(shape), tuple(stride))


def legacy_bytes(value, encoding):
    if encoding!='latin1' or not isinstance(value,str) or len(value)>32*1024**2:
        raise ValueError('unsupported legacy byte encoding')
    return value.encode('latin1')


class Metadata(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict': return OrderedDict
        if module == '_codecs' and name == 'encode': return legacy_bytes
        if module == 'torch' and name in ['FloatStorage','DoubleStorage','HalfStorage','BFloat16Storage','LongStorage','IntStorage','ByteStorage']:
            return name
        if module == 'torch._utils' and name in ['_rebuild_tensor_v2','_rebuild_tensor']:
            return tensor
        if (module, name) in [('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype')]:
            return Ignored
        raise ValueError('unsupported checkpoint pickle global: '+module+'.'+name)

    def persistent_load(self, value):
        if not isinstance(value,tuple) or len(value)!=5 or value[0]!='storage':
            raise ValueError('unsupported checkpoint storage reference')
        _,kind,key,location,count=value
        if not isinstance(key,str) or not key.isdigit() or type(count) is not int or count<0:
            raise ValueError('unsafe checkpoint storage')
        return Storage(kind,key,count)


DTYPES={'FloatStorage':('F32',4),'HalfStorage':('F16',2),'BFloat16Storage':('BF16',2)}


def metadata(archive):
    names=[n for n in archive.namelist() if n.endswith('/data.pkl')]
    if len(names)!=1: raise ValueError('one checkpoint data.pkl required')
    info=archive.getinfo(names[0])
    if info.file_size>32*1024**2: raise ValueError('checkpoint metadata exceeds 32 MiB')
    value=Metadata(io.BytesIO(archive.read(info))).load()
    if not isinstance(value,dict) or not isinstance(value.get('model'),dict) or not isinstance(value.get('config'),dict):
        raise ValueError('not a System One state-dict checkpoint')
    prefix=names[0][:-len('data.pkl')]
    if prefix+'byteorder' in archive.namelist() and archive.read(prefix+'byteorder')!=b'little':
        raise ValueError('only little-endian checkpoints supported')
    return value,prefix


def convert(checkpoint, destination, tokenizer, *, expected_step=None, require_done_log=None):
    checkpoint,destination,tokenizer=map(Path,(checkpoint,destination,tokenizer))
    if destination.exists(): raise FileExistsError('initialization output already exists')
    if require_done_log is None: raise ValueError('a completed parent log is required')
    rows=[]
    with Path(require_done_log).open() as f:
        for line in f:
            try: row=json.loads(line)
            except ValueError: continue
            if isinstance(row,dict): rows.append(row)
    last_start=max((i for i,r in enumerate(rows) if r.get('event') in ['start','resume']),default=-1)
    terminals=[(i,r) for i,r in enumerate(rows) if r.get('event')=='done']
    if not terminals or terminals[-1][0] <= last_start: raise ValueError('parent run has not completed; no conversion performed')
    final=terminals[-1][1]['step']
    if expected_step is not None and final!=expected_step: raise ValueError('parent completion step differs')
    before=checkpoint.stat()
    with zipfile.ZipFile(checkpoint) as z:
        value,prefix=metadata(z)
        if value.get('step')!=final: raise ValueError('checkpoint is not the completed parent step')
        header={};ranges=[];cursor=0
        for name,t in value['model'].items():
            if not isinstance(name,str) or not isinstance(t,Tensor) or t.storage.kind not in DTYPES:
                raise ValueError('unsupported model tensor')
            dtype,width=DTYPES[t.storage.kind];elements=math.prod(t.shape)
            if any(type(n) is not int or n<1 for n in t.shape) or len(t.shape)!=len(t.stride):
                raise ValueError('invalid model shape')
            stride=1
            for size,s in reversed(list(zip(t.shape,t.stride))):
                if size>1 and s!=stride: raise ValueError('noncontiguous model weights unsupported')
                stride*=size
            if type(t.offset) is not int or t.offset<0 or t.offset+elements>t.storage.count:
                raise ValueError('tensor exceeds storage')
            member=prefix+'data/'+t.storage.key;info=z.getinfo(member)
            if info.file_size!=t.storage.count*width: raise ValueError('storage size differs')
            count=elements*width;header[name]={'dtype':dtype,'shape':list(t.shape),'data_offsets':[cursor,cursor+count]}
            ranges.append((name,member,t.offset*width,count));cursor+=count
        if shutil.disk_usage(destination.parent).free < cursor+256*1024**2: raise RuntimeError('insufficient space for converted weights')
        encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
        destination.mkdir(mode=0o700);target=destination/'model.safetensors';digests={}
        try:
            with target.open('xb') as out:
                out.write(struct.pack('<Q',len(encoded)));out.write(encoded)
                for name,member,offset,count in ranges:
                    digest=hashlib.sha256()
                    with z.open(member) as src:
                        if offset: src.seek(offset)
                        remaining=count
                        while remaining:
                            chunk=src.read(min(2*1024**2,remaining))
                            if not chunk: raise ValueError('truncated tensor storage')
                            out.write(chunk);digest.update(chunk);remaining-=len(chunk)
                    digests[name]=digest.hexdigest()
            after=checkpoint.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise ValueError('parent checkpoint changed during conversion')
            record={'schema':'sys1-mlx-initialization-v1','parent_completed':True,'parent_step':final,
                    'parent_checkpoint':str(checkpoint.resolve()),'parent_checkpoint_identity':file_identity(checkpoint),
                    'parent_log_identity':file_identity(require_done_log),'config':value['config'],
                    'tokenizer_identity':file_identity(tokenizer),'model_identity':file_identity(target),
                    'tensor_sha256':digests,'optimizer':'reset identically in every arm','torch_imported':False,
                    'method':'restricted metadata parser; raw tensor bytes copied without dtype conversion'}
            (destination/'initialization.json').write_text(json.dumps(record,indent=2)+'\n')
            for p in destination.iterdir():p.chmod(0o600)
            return record
        except BaseException:
            shutil.rmtree(destination);raise
