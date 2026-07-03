import torch

from retrain.backends.local import batch as local_batch


def test_policy_batch_pads_rows_and_builds_attention_mask() -> None:
    batch = local_batch.policy(
        [[11, 12, 13], [21]],
        [[0.0, -0.1, -0.2], [0.0]],
        [[0.0, 1.0, 2.0], [0.0]],
        device="cpu",
    )

    assert batch.input_ids.dtype == torch.long
    assert batch.input_ids.tolist() == [[11, 12, 13], [21, 0, 0]]
    assert batch.old_logprobs.dtype == torch.float32
    torch.testing.assert_close(
        batch.old_logprobs,
        torch.tensor([[0.0, -0.1, -0.2], [0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        batch.advantages,
        torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]),
    )
    assert batch.attention_mask.dtype == torch.bool
    assert batch.attention_mask.tolist() == [[True, True, True], [True, False, False]]


def test_sft_batch_uses_token_lengths_for_mask() -> None:
    batch = local_batch.sft(
        [[1, 2], [3, 4, 5]],
        [[0.0, 1.0], [0.0, 0.5, 0.0]],
        device="cpu",
    )

    assert batch.input_ids.tolist() == [[1, 2, 0], [3, 4, 5]]
    torch.testing.assert_close(
        batch.advantages,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]]),
    )
    assert batch.attention_mask.tolist() == [[True, True, False], [True, True, True]]


def test_echo_batch_pads_echo_advantages_and_counts() -> None:
    batch = local_batch.echo(
        [[1, 2, 3], [4, 5]],
        [[0.0, -0.1, -0.2], [0.0, -0.4]],
        [[0.0, 0.2, 0.3], [0.0, 0.5]],
        [[0.0, 1.0, 0.0], [0.0, 0.25]],
        [2, 1],
        device="cpu",
    )

    assert batch.input_ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    torch.testing.assert_close(
        batch.old_logprobs,
        torch.tensor([[0.0, -0.1, -0.2], [0.0, -0.4, 0.0]]),
    )
    torch.testing.assert_close(
        batch.advantages,
        torch.tensor([[0.0, 0.2, 0.3], [0.0, 0.5, 0.0]]),
    )
    torch.testing.assert_close(
        batch.echo_advantages,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.25, 0.0]]),
    )
    torch.testing.assert_close(batch.echo_counts, torch.tensor([2.0, 1.0]))
    assert batch.attention_mask.tolist() == [[True, True, True], [True, True, False]]
