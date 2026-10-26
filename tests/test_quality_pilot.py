import importlib.util
from pathlib import Path

import pytest


def test_frozen_validation_covers_reservoir_without_padding_or_order_dependence():
    path=Path(__file__).resolve().parents[1]/'scripts/compare_quality_mixture.py'
    spec=importlib.util.spec_from_file_location('quality_pilot',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    packs={'short':range(17),'long':range(512)}
    selected=module.validation_indices(packs,512)
    assert set(selected['short'])==set(range(17))
    assert set(selected['long'])==set(range(512))
    assert selected==module.validation_indices(dict(reversed(list(packs.items()))),512)
    default=module.validation_indices(packs,16)
    assert all(len(set(indices))==16 for indices in default.values())
    assert default==module.validation_indices(packs,16)
    for count in [0,513]:
        with pytest.raises(ValueError):module.validation_indices(packs,count)
