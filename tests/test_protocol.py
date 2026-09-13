import json

import numpy as np
import polars as pl
import pytest
import torch

from careseq.baselines import NGram
from careseq.data import EVENTS, N_CLASSES, generate, split_patients, windows
from careseq.metrics import classification_metrics, paired_patient_bootstrap
from careseq.model import CareTransformer

torch.set_num_threads(2)


def small_data():
    frame = generate(30, 4)
    split = split_patients(frame["patient_id"].to_list(), 7)
    return frame, split, {s: windows(frame, ids, 12, 3) for s, ids in split.items()}


def test_generation_reproducible_and_chronological():
    a = generate(20, 1)
    assert a.equals(generate(20, 1))
    assert not a.equals(generate(20, 2))
    for patient in a.partition_by("patient_id"):
        dates = patient["event_date"].to_list()
        assert all(left < right for left, right in zip(dates, dates[1:]))
        assert patient["synthetic"].all()


def test_patient_splits_disjoint_complete():
    frame, splits, _ = small_data()
    sets = {s: set(ids) for s, ids in splits.items()}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
    assert set.union(*sets.values()) == set(frame["patient_id"])


def test_windows_exclude_future_and_preserve_target_alignment():
    frame, _, data = small_data()
    sequences = {p["patient_id"][0]: p["event_id"].to_list() for p in frame.partition_by("patient_id")}
    for x, y, length, patient, position in zip(*[data["test"][k] for k in ["x", "y", "lengths", "patients", "target_position"]]):
        source = sequences[patient]
        assert x[:length].tolist() == source[max(0, position-12):position]
        assert y == source[position] - 1
        assert (x[length:] == 0).all()


def test_padding_does_not_change_prediction():
    torch.manual_seed(2)
    model = CareTransformer(max_context=8, dropout=0).eval()
    with torch.inference_mode():
        short = model(torch.tensor([[1, 2, 3]]), torch.tensor([3]))
        padded = model(torch.tensor([[1, 2, 3, 0, 0, 0]]), torch.tensor([3]))
    assert torch.allclose(short, padded, atol=1e-6)


def test_model_gradient_and_output_classes():
    model = CareTransformer(max_context=8)
    logits = model(torch.tensor([[1, 2, 3], [4, 5, 0]]), torch.tensor([3, 2]))
    assert logits.shape == (2, N_CLASSES)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert torch.isfinite(model.head.weight.grad).all()
    assert model.head.weight.grad.abs().sum() > 0
    assert torch.equal(model.token_embedding.weight.grad[0], torch.zeros(32))


def test_baseline_probabilities_and_unseen_context_backoff():
    _, _, data = small_data()
    model = NGram(2).fit(data["train"])
    test = {k: v.copy() for k, v in data["test"].items()}
    test["x"][:] = 999  # Contexte absent : repli sur fréquence globale.
    probabilities = model.probabilities(test)
    assert np.allclose(probabilities.sum(1), 1)
    assert (probabilities > 0).all()
    assert np.allclose(probabilities, NGram(0).fit(data["train"]).probabilities(test))


def test_metrics_perfect_predictions():
    target = np.arange(N_CLASSES)
    metrics = classification_metrics(target, np.eye(N_CLASSES))
    assert metrics["accuracy"] == metrics["macro_f1"] == metrics["top3_accuracy"] == 1
    assert metrics["nll"] == 0


def test_bootstrap_identical_models_zero_difference():
    target = np.array([0, 1, 2, 3])
    probabilities = np.eye(N_CLASSES)[target]
    result = paired_patient_bootstrap(target, probabilities, probabilities, np.array(["a", "a", "b", "b"]), repeats=50)
    assert result["delta_accuracy"] == 0
    assert result["ci95"] == [0, 0]


def test_checkpoint_reload_preserves_predictions(tmp_path):
    model = CareTransformer(max_context=8).eval()
    checkpoint = {"state_dict": model.state_dict(), "architecture": {"max_context": 8}}
    torch.save(checkpoint, tmp_path / "model.pt")
    loaded = torch.load(tmp_path / "model.pt", weights_only=True)
    other = CareTransformer(**loaded["architecture"]).eval()
    other.load_state_dict(loaded["state_dict"])
    with torch.inference_mode():
        x, lengths = torch.tensor([[1, 2, 3]]), torch.tensor([3])
        assert torch.equal(model(x, lengths), other(x, lengths))
