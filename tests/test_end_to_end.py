import json

from careseq.data import prepare
from careseq.inference import infer_events
from careseq.report import render
from careseq.train import run_experiment


def test_mini_experiment_reload_and_report(tmp_path):
    config = {"patients": 60, "data_seed": 2026, "split_seed": 31, "model_seeds": [7],
              "max_context": 8, "min_context": 2, "d_model": 8, "heads": 2, "layers": 1,
              "feedforward": 16, "dropout": 0.1, "batch_size": 128, "epochs": 1,
              "patience": 2, "learning_rate": 0.001, "threads": 2, "bootstrap_repeats": 20}
    data, manifest = prepare(tmp_path, config)
    result = run_experiment(tmp_path, data, manifest, config)
    render(tmp_path)
    prediction = infer_events(tmp_path, ["GENERALISTE", "BIOLOGIE", "CARDIOLOGIE"])
    assert result["selected_seed"] == 7
    assert set(result["metrics"]) == {"Frequence", "Markov_1", "Markov_2", "Transformer"}
    assert len(prediction["top3"]) == 3
    assert prediction["synthetic_only"]
    assert (tmp_path / "reports/figures/confusion.png").stat().st_size > 0
    assert "Données entièrement synthétiques" in (tmp_path / "reports/rapport.html").read_text(encoding="utf-8")
