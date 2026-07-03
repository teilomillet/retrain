from __future__ import annotations

import os

import pytest
import torch

from retrain.backends.local import state as local_state


class _StateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.base_weight = torch.nn.Parameter(torch.tensor([3.0]))


class _SaveModel:
    def __init__(self) -> None:
        self.saved_to: list[str] = []

    def save_pretrained(self, path: str) -> None:
        self.saved_to.append(path)
        with open(os.path.join(path, "adapter_model.bin"), "wb") as handle:
            handle.write(b"fake")


def test_lora_state_dict_can_share_or_clone_storage() -> None:
    model = _StateModel()

    live = local_state.lora_state_dict(model, clone=False)
    snapshot = local_state.lora_state_dict(model, clone=True)

    assert list(live) == ["lora_weight"]
    assert live["lora_weight"].data_ptr() == model.lora_weight.data.data_ptr()
    assert snapshot["lora_weight"].data_ptr() != model.lora_weight.data.data_ptr()


def test_resolve_adapter_dir_accepts_direct_weight_directory(tmp_path) -> None:
    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "adapter_model.bin").write_bytes(b"fake")

    assert local_state.resolve_adapter_dir("/unused", direct) == os.fspath(direct)


def test_load_adapter_weights_reports_missing_weight_files(tmp_path) -> None:
    adapter = tmp_path / "checkpoint"
    adapter.mkdir()

    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        local_state.load_adapter_weights(os.fspath(adapter), "cpu")


def test_save_model_creates_named_adapter_directory(tmp_path) -> None:
    model = _SaveModel()

    save_dir = local_state.save_model(model, path=os.fspath(tmp_path), name="step_1")

    assert save_dir == os.fspath(tmp_path / "step_1")
    assert model.saved_to == [save_dir]
    assert (tmp_path / "step_1" / "adapter_model.bin").is_file()
