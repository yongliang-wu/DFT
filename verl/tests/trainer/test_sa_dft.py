from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl.trainer.fsdp_dft_trainer import FSDPSFTTrainer


class DummyTokenizer:
    def __init__(self):
        self._mapping = {
            0: "Ġthe",
            1: "Ġ42",
            2: "Ġmatrix",
            3: "Ġ,",
        }

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, list):
            return [self._mapping[i] for i in ids]
        return self._mapping[ids]


def _build_trainer(sa_dft_config):
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.config = SimpleNamespace(sa_dft=sa_dft_config)
    trainer.tokenizer = DummyTokenizer()
    trainer._sa_dft_runtime_config = None
    trainer._sa_dft_weight_cache = {}
    return trainer


def test_sa_dft_weights_disabled():
    trainer = _build_trainer({"enable": False})
    token_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    reference = torch.ones(3, dtype=torch.float32)
    weights = trainer._compute_semantic_weights(token_ids, reference)
    assert torch.allclose(weights, torch.ones_like(reference))


def test_sa_dft_weights_with_simple_heuristics():
    trainer = _build_trainer(
        {"enable": True, "numeric_weight": 0.2, "stopword_weight": 0.9, "default_weight": 0.7}
    )
    token_ids = torch.tensor([1, 0, 2, 3], dtype=torch.long)
    reference = torch.ones(4, dtype=torch.float32)
    weights = trainer._compute_semantic_weights(token_ids, reference)

    assert weights.shape == reference.shape
    assert weights[0].item() == pytest.approx(0.2)  # numeric token
    assert weights[1].item() == pytest.approx(0.9)  # stopword token
    assert weights[2].item() == pytest.approx(0.7)  # default token
    assert weights[3].item() == pytest.approx(0.9)  # punctuation treated as stopword


def test_sa_dft_respects_additional_stopwords():
    trainer = _build_trainer({"enable": True, "additional_stopwords": ["matrix"]})
    token_ids = torch.tensor([2], dtype=torch.long)
    reference = torch.ones(1, dtype=torch.float32)
    weights = trainer._compute_semantic_weights(token_ids, reference)

    runtime_config = trainer._get_sa_dft_runtime_config()
    assert weights.item() == pytest.approx(runtime_config["stopword_weight"])
