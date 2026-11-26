"""Mock-based tests for training interfaces, without framework/model loading."""

from unittest.mock import Mock
from types import ModuleType
import sys

from src.training.sft import SFTTrainer


def test_sft_trainer_delegates_to_trl(monkeypatch) -> None:
    import src.training.sft as sft_module

    fake_model, fake_tokenizer, fake_result = object(), object(), object()
    loader = Mock(return_value=(fake_model, fake_tokenizer))
    trainer = Mock()
    trainer.train.return_value = fake_result
    trainer_cls = Mock(return_value=trainer)
    monkeypatch.setattr(sft_module, "load_training_components", loader)
    monkeypatch.setattr(sft_module, "training_arguments", lambda cfg, out: object())
    fake_trl = ModuleType("trl")
    fake_trl.SFTTrainer = trainer_cls
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    result = SFTTrainer().train({"output_dir": "out", "lora": {}}, "local-model", [])
    assert result is fake_result
    trainer_cls.assert_called_once()
