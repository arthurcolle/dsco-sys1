"""Read-only loopback source for existing cleaned JSONL; never copies a corpus."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import tempfile
import urllib.parse

from sys1.model.corpus_manifest import file_identity,load_manifest

CHUNK_BYTES=256*1024


def publish(corpus,tokenizer,output):
    output=Path(output)
    if output.exists():raise FileExistsError('stream index output already exists')
    sources,parent_id,manifest=load_manifest(corpus)
    if len(sources)>32:raise ValueError('at most 32 sources supported')
    public={};private={}
    for name,entry in sources.items():
        public[name]={}
        for split,paths in entry.items():
            public[name][split]=[]
            for i,path in enumerate(paths):
                hashes=[]
                with path.open('rb') as f:
                    while chunk:=f.read(CHUNK_BYTES):hashes.append(hashlib.sha256(chunk).hexdigest())
                    f.seek(-1,2)
                    if f.read(1)!=b'\n':raise ValueError('stream JSONL files must end in newline')
                if not path.name.endswith('.jsonl'):raise ValueError('publish only existing cleaned JSONL files')
                resource=hashlib.sha256((name+split+str(i)+str(path)).encode()).hexdigest()[:32]
                stat=path.stat();private[resource]={'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'inode':stat.st_ino}
                public[name][split].append({'resource':'chunk/'+resource,'bytes':stat.st_size,'chunks':hashes,'ends_newline':True})
    tok=file_identity(tokenizer)
    if manifest.get('tokenizer_identity',{}).get('sha256')!=tok['sha256']:raise ValueError('clean corpus tokenizer identity differs')
    value={'schema':'sys1-http-corpus-v1','chunk_bytes':CHUNK_BYTES,'sources':public,
           'objective_weights':manifest['objective_weights'],'tokenizer_sha256':tok['sha256'],
           'parent_manifest_sha256':parent_id,'privacy':'serve only on loopback through authenticated fleet tunnel',
           'evaluation_limitations':manifest.get('audit',{}).get('unmeasured',[])}
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
    if len(raw)>2*1024**2:raise ValueError('stream index exceeds 2 MiB')
    output.mkdir(mode=0o700)
    (output/'manifest.json').write_bytes(raw)
    (output/'server-index.json').write_text(json.dumps(private)+'\n')
    record={'manifest_sha256':hashlib.sha256(raw).hexdigest(),'parent_manifest_sha256':parent_id,
            'tokenizer_sha256':tok['sha256'],'dataset_bytes_copied':0,'index_bytes':len(raw),
            'source_bytes':sum(v['bytes'] for v in private.values()),'training_launched':False}
    (output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    for p in output.iterdir():p.chmod(0o600)
    return record


def server(index,port=8955):
    index=Path(index);raw=(index/'manifest.json').read_bytes();files=json.loads((index/'server-index.json').read_text())
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            route=urllib.parse.urlsplit(self.path)
            if route.query or route.fragment:self.send_error(400);return
            if route.path=='/manifest.json':data=raw
            else:
                match=re.fullmatch(r'/chunk/([0-9a-f]{32})/([0-9]{1,12})',route.path)
                if not match or match[1] not in files:self.send_error(404);return
                entry=files[match[1]];number=int(match[2]);offset=number*CHUNK_BYTES
                if offset>=entry['bytes']:self.send_error(404);return
                p=Path(entry['path'])
                if p.is_symlink():self.send_error(409);return
                try:
                    with p.open('rb') as f:
                        st=os.fstat(f.fileno())
                        if (st.st_size,st.st_mtime_ns,st.st_ino)!=(entry['bytes'],entry['mtime_ns'],entry['inode']):
                            self.send_error(409);return
                        f.seek(offset);data=f.read(min(CHUNK_BYTES,entry['bytes']-offset))
                except OSError:self.send_error(503);return
            self.send_response(200);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def set_policy(path,manifest_sha,values,version):
    if not isinstance(manifest_sha,str) or not re.fullmatch(r'[0-9a-f]{64}',manifest_sha):raise ValueError('manifest SHA-256 required')
    if type(version) is not int or version<1:raise ValueError('policy version must be positive integer')
    if not isinstance(values,dict) or not values or any(type(v) not in [int,float] or not math.isfinite(v) or v<0 for v in values.values()):
        raise ValueError('nonnegative finite source weights required')
    total=math.fsum(values.values())
    if not math.isfinite(total) or total<=0:raise ValueError('invalid source weights')
    normalized={k:v/total for k,v in values.items()}
    if any(normalized[k]==0 and values[k]>0 for k in values):raise ValueError('weight normalization underflow')
    path=Path(path)
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('stream_manifest_sha256')!=manifest_sha or set(old.get('weights',{}))!=set(values) or version<=old.get('version',0):
            raise ValueError('policy identities/sources must match and version must increase')
    record={'schema':'sys1-stream-policy-v1','stream_manifest_sha256':manifest_sha,'version':version,'weights':normalized}
    payload=json.dumps(record,indent=2)+'\n'
    if len(payload.encode())>65536:raise ValueError('policy exceeds 64 KiB')
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    build=sub.add_parser('publish');build.add_argument('--corpus',required=True);build.add_argument('--tokenizer',required=True);build.add_argument('--output',required=True)
    serve=sub.add_parser('serve');serve.add_argument('--index',required=True);serve.add_argument('--port',type=int,default=8955)
    policy=sub.add_parser('policy');policy.add_argument('--path',required=True);policy.add_argument('--manifest-sha256',required=True);policy.add_argument('--weights',required=True);policy.add_argument('--version',required=True,type=int)
    args=p.parse_args()
    if args.command=='publish':print(json.dumps(publish(args.corpus,args.tokenizer,args.output)))
    elif args.command=='serve':server(args.index,args.port).serve_forever()
    else:print(json.dumps(set_policy(args.path,args.manifest_sha256,json.loads(args.weights),args.version)))


if __name__=='__main__':main()
