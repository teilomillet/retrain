from __future__ import annotations

from types import SimpleNamespace

import torch

from retrain.backends.local import logprobs as local_logprobs


class _ProbeModel(torch.nn.Module):
    def __init__(self, *, supports_logits_to_keep: bool = True) -> None:
        super().__init__()
        self.supports_logits_to_keep = supports_logits_to_keep

    def forward(self, **kwargs):
        if not self.supports_logits_to_keep:
            raise TypeError("logits_to_keep is unsupported")
        input_ids = kwargs["input_ids"]
        logits_to_keep = int(kwargs["logits_to_keep"])
        batch = int(input_ids.shape[0])
        return SimpleNamespace(logits=torch.zeros(batch, logits_to_keep, 4))


class _DenseOwner:
    train_logprob_chunk_size = 0
    train_selective_suffix_logits = False

    def __init__(self) -> None:
        self.train_model = object()
        self._loss_path_counts: dict[str, int] = {}

    def _selective_suffix_token_logprobs(self, input_ids, attention_mask, target_mask):
        return None

    def _selective_hidden_token_logprobs(self, input_ids, attention_mask, target_mask):
        return None

    def _liger_fused_linear_ce_loss(self):
        return None


def test_supports_train_logits_to_keep_caches_probe_result() -> None:
    owner = SimpleNamespace(
        train_model=_ProbeModel(supports_logits_to_keep=True),
        _train_logits_to_keep_supported=None,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    assert local_logprobs.supports_train_logits_to_keep(
        owner,
        input_ids,
        attention_mask,
    )
    assert owner._train_logits_to_keep_supported is True

    owner.train_model.supports_logits_to_keep = False
    assert local_logprobs.supports_train_logits_to_keep(
        owner,
        input_ids,
        attention_mask,
    )


def test_shifted_token_logprobs_dense_path_gathers_target_tokens() -> None:
    owner = _DenseOwner()
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    def fake_forward_logits(model, ids, mask):
        _ = model, ids, mask
        return torch.tensor(
            [
                [
                    [4.0, 1.0, 0.0],
                    [0.0, 2.0, 5.0],
                    [9.0, 9.0, 9.0],
                ]
            ],
            dtype=torch.float32,
        )

    actual = local_logprobs.shifted_token_logprobs(
        owner,
        input_ids,
        attention_mask,
        forward_logits_fn=fake_forward_logits,
    )

    logits = fake_forward_logits(None, input_ids, attention_mask)[:, :-1]
    expected = (
        torch.log_softmax(logits, dim=-1)
        .gather(
            2,
            input_ids[:, 1:].unsqueeze(2),
        )
        .squeeze(2)
    )
    assert torch.allclose(actual, expected)
    assert owner._loss_path_counts == {"dense_logprob": 1}
