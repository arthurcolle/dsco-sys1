import io
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pytest
from sys1.mlx.legacy_weights import Storage,Tensor,tensor,convert,metadata


class TensorPickle:
    def __init__(self,storage):self.storage=storage
    def __reduce__(self):return tensor,(self.storage,0,(2,3),(3,1),False,None)


class Writer(pickle.Pickler):
    def persistent_id(self,value):
        if isinstance(value,Storage):return ('storage',value.kind,value.key,'mps:0',value.count)


def fixture(root,done=True):
    storage=Storage('FloatStorage','0',6)
    checkpoint={'step':30,'config':{'vocab_size':2},'model':{'embed.weight':TensorPickle(storage)}}
    b=io.BytesIO();Writer(b,protocol=2).dump(checkpoint)
    raw=b.getvalue().replace(b'csys1.mlx.legacy_weights\ntensor\n',b'ctorch._utils\n_rebuild_tensor_v2\n')
    cp=root/'checkpoint.pt';values=np.arange(6,dtype=np.float32).reshape(2,3)
    with zipfile.ZipFile(cp,'w') as z:
        z.writestr('checkpoint/data.pkl',raw);z.writestr('checkpoint/byteorder','little')
        z.writestr('checkpoint/data/0',values.tobytes())
    tok=root/'tokenizer.json';tok.write_text('{}')
    log=root/'log.jsonl';log.write_text(json.dumps({'event':'done' if done else 'train','step':30})+'\n')
    return cp,tok,log,values


def test_raw_model_conversion_is_exact_and_torch_free(tmp_path):
    import mlx.core as mx
    cp,tok,log,values=fixture(tmp_path)
    r=convert(cp,tmp_path/'initial',tok,expected_step=30,require_done_log=log)
    result=mx.load(str(tmp_path/'initial/model.safetensors'))
    np.testing.assert_array_equal(np.array(result['embed.weight']),values)
    assert r['parent_completed'] and r['torch_imported'] is False
    with pytest.raises(FileExistsError):convert(cp,tmp_path/'initial',tok,expected_step=30,require_done_log=log)


def test_parent_completion_is_required_before_any_output(tmp_path):
    cp,tok,log,_=fixture(tmp_path,done=False)
    with pytest.raises(ValueError,match='not completed'):convert(cp,tmp_path/'initial',tok,expected_step=30,require_done_log=log)
    assert not (tmp_path/'initial').exists()


def test_untrusted_pickle_globals_are_rejected(tmp_path):
    cp=tmp_path/'bad.pt'
    with zipfile.ZipFile(cp,'w') as z:z.writestr('bad/data.pkl',b'cos\nsystem\n(S"echo unsafe"\ntR.')
    with zipfile.ZipFile(cp) as z:
        with pytest.raises(ValueError,match='unsupported'):metadata(z)
