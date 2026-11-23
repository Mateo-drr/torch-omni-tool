"""
Tests for the training scaffold: dataset, loops, utils and the training script
"""

import copy
import dataclasses
import runpy
from collections import defaultdict
from typing import Any

import pytest
import torch
from torch import nn

from torch_omni_tool import loops, utils
from torch_omni_tool.config import Config
from torch_omni_tool.datasets import CustomDataset, make_dl
from torch_omni_tool.model import SampleNet

pytestmark = pytest.mark.filterwarnings(
    "ignore:To copy construct from a tensor",  # CustomDataset re-wraps a tensor
    "ignore:.*pin_memory.*",  # no accelerator in CI
)


def _config(**overrides: Any) -> Config:
    """
    Build a small CPU config for tests
    Args:
        overrides (Any): Fields to change on top of the test defaults
    Returns:
        Config: Config with threads disabled, no workers and wandb off
    """
    base = {"threads": None, "num_workers": 0, "wb": False, "num_epochs": 2, "half_p": False}
    return dataclasses.replace(Config(), **{**base, **overrides})


def _loader(batches: int = 2) -> list[dict[str, torch.Tensor]]:
    """
    Build a fixed list of batches shaped like the CustomDataset samples
    Args:
        batches (int): Number of batches
    Returns:
        list[dict[str, torch.Tensor]]: Batches with the input under the data key
    """
    gen = torch.Generator().manual_seed(0)
    return [{"data": torch.randn(4, 8, 16, generator=gen)} for _ in range(batches)]


# dataset


def test_dataset_returns_data_samples() -> None:
    dataset = CustomDataset(_config())
    assert len(dataset) == 1
    assert dataset[0]["data"].shape == (8, 16)


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_make_dl_batches_the_dataset(split: str) -> None:
    batch = next(iter(make_dl(_config(batch=4), split)))  # type: ignore[arg-type]
    assert batch["data"].shape == (1, 8, 16)


def test_make_dl_rejects_unknown_split() -> None:
    with pytest.raises(NotImplementedError):
        make_dl(_config(), "holdout")  # type: ignore[arg-type]


# loops


def test_run_model_returns_loss_and_tensors() -> None:
    model = SampleNet()
    out = loops.run_model(model, _loader(1)[0], nn.MSELoss(), _config())
    assert set(out) == {"loss", "data", "output"}
    assert out["output"].shape == (4, 8, 16)
    assert out["loss"].ndim == 0


@pytest.mark.parametrize("half_p", [False, True], ids=["fp32", "amp"])
def test_train_loop_reduces_the_loss(half_p: bool) -> None:
    torch.manual_seed(0)
    config = _config(half_p=half_p, lr=1e-2)
    model = SampleNet()
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler(device="cpu", enabled=half_p)
    data = _loader(4)
    metrics: dict[str, dict[str, Any]] = {"train": {}, "valid": {}}
    losses = []
    for epoch in range(5):
        loops.train_loop(model, data, nn.MSELoss(), optim, scaler, metrics, config, epoch)  # type: ignore[arg-type]
        losses.append(metrics["train"]["loss"])
    assert losses[-1] < losses[0]


def test_amp_clips_unscaled_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    # regression: clipping used to run on gradients still multiplied by the loss scale
    torch.manual_seed(0)
    model = SampleNet()
    batch = _loader(1)
    reference = copy.deepcopy(model)
    nn.MSELoss()(reference(batch[0]["data"]), batch[0]["data"]).backward()
    expected = torch.norm(
        torch.stack([p.grad.norm() for p in reference.parameters() if p.grad is not None])
    )

    seen: list[float] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy(params: Any, max_norm: float, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Record the gradient norm at clipping time
        Args:
            params (Any): Parameters passed to clip_grad_norm_
            max_norm (float): Clipping threshold
            args (Any): Other positional arguments
            kwargs (Any): Other keyword arguments
        Returns:
            torch.Tensor: Total norm returned by the real function
        """
        norm = real_clip(params, max_norm, *args, **kwargs)
        seen.append(float(norm))
        return norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    config = _config(half_p=True, grad_clip=1e9)
    scaler = torch.amp.GradScaler(device="cpu")
    assert scaler.get_scale() > 1000
    optim = torch.optim.SGD(model.parameters(), lr=0.0)
    metrics: dict[str, dict[str, Any]] = {"train": {}, "valid": {}}
    loops.train_loop(model, batch, nn.MSELoss(), optim, scaler, metrics, config, 0)  # type: ignore[arg-type]
    assert len(seen) == 1
    # autocast runs in bfloat16, so allow a loose margin; scaled gradients would be ~65536x off
    assert 0.5 < seen[0] / float(expected) < 2.0


def test_train_loop_steps_the_scheduler_once_per_epoch() -> None:
    model = SampleNet()
    optim = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5)
    metrics: dict[str, dict[str, Any]] = {"train": {}, "valid": {}}
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    for epoch in range(3):
        loops.train_loop(
            model, _loader(2), nn.MSELoss(), optim, scaler, metrics, _config(), epoch, scheduler  # type: ignore[arg-type]
        )
    assert optim.param_groups[0]["lr"] == pytest.approx(0.125)


@pytest.mark.parametrize("half_p", [False, True], ids=["fp32", "amp"])
def test_eval_loop_does_not_change_the_model(half_p: bool) -> None:
    model = SampleNet()
    before = copy.deepcopy(model.state_dict())
    metrics: dict[str, dict[str, Any]] = {"train": {}, "valid": {}}
    loops.eval_loop(model, _loader(2), nn.MSELoss(), "valid", metrics, _config(half_p=half_p))  # type: ignore[arg-type]
    assert not model.training
    assert "loss" in metrics["valid"]
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.xfail(strict=True, reason="known issue: eval_loop always logs under 'valid'")
def test_eval_loop_logs_under_its_name() -> None:
    metrics: dict[str, dict[str, Any]] = {"train": {}, "valid": {}, "test": {}}
    loops.eval_loop(SampleNet(), _loader(1), nn.MSELoss(), "test", metrics, _config())  # type: ignore[arg-type]
    assert "loss" in metrics["test"]


@pytest.mark.xfail(
    strict=True, reason="known issue: the loss gets (data, output) instead of (output, data)"
)
def test_loss_receives_prediction_then_target() -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def criterion(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Record the arguments of the loss
        Args:
            pred (torch.Tensor): First argument, expected to be the prediction
            target (torch.Tensor): Second argument, expected to be the target
        Returns:
            torch.Tensor: Mean squared error
        """
        calls.append((pred, target))
        return ((pred - target) ** 2).mean()

    model = SampleNet()
    batch = _loader(1)[0]
    out = loops.run_model(model, batch, criterion, _config())  # type: ignore[arg-type]
    assert calls[0][0] is out["output"]


# utils


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00m"), (59, "00m"), (60, "01m"), (3599, "59m"), (3600, "01:00h"), (5400, "01:30h")],
)
def test_format_time(seconds: float, expected: str) -> None:
    assert utils.format_time(seconds) == expected


def test_format_metrics_converts_tensors() -> None:
    results: defaultdict[str, list[Any]] = defaultdict(list)
    utils.format_metrics({"loss": torch.tensor(0.5), "acc": 1.0}, results)
    utils.format_metrics({"loss": torch.tensor(1.5), "acc": 0.0}, results)
    assert results == {"loss": [0.5, 1.5], "acc": [1.0, 0.0]}


def test_epoch_results_averages_each_metric() -> None:
    out = utils.epoch_results({"loss": [1.0, 3.0]}, {"train": {}}, "train")
    assert out == {"train": {"loss": 2.0}}


def test_count_params(capsys: pytest.CaptureFixture[str]) -> None:
    model = nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    utils.count_params(model)
    assert capsys.readouterr().out.split() == [
        "Total",
        "parameters:",
        "8",
        "Trainable",
        "parameters:",
        "6",
    ]


@pytest.mark.parametrize(
    ("losses", "best_epoch"), [([3.0, 2.0, 1.0], 2), ([1.0, 2.0, 3.0], 0), ([2.0, 1.0, 1.5], 1)]
)
def test_finish_epoch_keeps_the_best_model(losses: list[float], best_epoch: int) -> None:
    config = _config(num_epochs=len(losses))
    timings = {"start": 0.0, "epoch_start": 0.0}
    best: dict[str, Any] = {}
    models = []
    for epoch, loss in enumerate(losses):
        model = SampleNet()
        models.append(model)
        metrics = {"train": {"loss": loss}, "valid": {"loss": loss}}
        best = utils.finish_epoch(epoch, metrics, timings, best, 0.1, model, config)
    assert best["loss"] == losses[best_epoch]
    expected = models[best_epoch].state_dict()
    assert best["model"] is not models[best_epoch]
    assert all(torch.equal(v, expected[k]) for k, v in best["model"].state_dict().items())


# training script


# runpy warns because other tests already imported the module, which is harmless here
@pytest.mark.filterwarnings("ignore:'torch_omni_tool.training' found in sys.modules")
def test_training_script_steps_the_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    # regression: the script built a scheduler but passed scheduler=None to the train loop
    from torch_omni_tool.config import config

    for key, value in {"threads": None, "num_workers": 0, "wb": False, "num_epochs": 2}.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setattr(config, "half_p", False)
    namespace = runpy.run_module("torch_omni_tool.training", run_name="__main__")
    scheduler = namespace["scheduler"]
    assert scheduler is not None and scheduler.last_epoch == 2
    assert namespace["optimizer"].param_groups[0]["lr"] < config.lr
