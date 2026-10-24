import gzip

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from sys1.model.pretrain import DocReader, heldout_windows


def tok():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, **{f"w{i}": i+2 for i in range(1000)}}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def test_reservoir_covers_late_ordered_categories_and_is_deterministic(tmp_path):
    path = tmp_path / "heldout.txt"
    path.write_text("\n".join(f"w{i}" for i in range(1000)) + "\n")
    metadata = {}
    sampled = heldout_windows([path], None, tok(), 1, 4, 32, "SEP", "reservoir", 1234, metadata)
    other = heldout_windows([path], None, tok(), 1, 4, 32, "SEP", "reservoir", 1234)
    assert torch.equal(sampled, other)
    assert metadata["complete_pass"] is True
    assert metadata["total_windows"] == 500
    assert len(set(metadata["selected_window_indices"])) == 32
    assert min(metadata["selected_window_indices"]) < 125
    assert max(metadata["selected_window_indices"]) >= 375
    changed = heldout_windows([path], None, tok(), 1, 4, 32, "SEP", "reservoir", 5678)
    assert not torch.equal(sampled, changed)


def test_short_validation_is_not_repeated_to_fill_sample_budget(tmp_path):
    path = tmp_path / "heldout.txt"
    path.write_text("w1\nw2\nw3\n")
    metadata = {}
    sampled = heldout_windows([path], None, tok(), 1, 4, 20, "SEP", "reservoir", 1, metadata)
    assert sampled.shape == (1, 4)
    assert metadata["tail_tokens"] == 2
    assert metadata["documents"] == 3


def test_finite_reader_stays_exhausted_and_supports_gzip(tmp_path):
    path = tmp_path / "heldout.txt.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("w1\nw2\n")
    reader = DocReader([path], "SEP", repeat=False)
    assert [reader.next_doc(), reader.next_doc()] == ["w1", "w2"]
    for _ in range(2):
        with pytest.raises(StopIteration):
            reader.next_doc()


def test_prefix_mode_remains_available_for_historical_recipes(tmp_path):
    path = tmp_path / "heldout.txt"
    path.write_text("\n".join(f"w{i}" for i in range(1000)) + "\n")
    metadata = {}
    selected = heldout_windows([path], None, tok(), 1, 4, 32, "SEP", "prefix", 0, metadata)
    assert selected[0].tolist() == [2, 1, 3, 1]
    assert metadata["selected_window_indices"] == list(range(32))
