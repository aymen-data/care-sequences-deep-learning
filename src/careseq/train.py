import copy
import json
import logging
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .baselines import NGram
from .data import EVENTS, write_json
from .metrics import classification_metrics, paired_patient_bootstrap
from .model import CareTransformer, architecture

LOG = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@torch.inference_mode()
def predict(model, data, batch_size=512):
    model.eval()
    results = []
    for start in range(0, len(data["x"]), batch_size):
        tokens = torch.from_numpy(data["x"][start:start+batch_size])
        lengths = torch.from_numpy(data["lengths"][start:start+batch_size])
        results.append(model(tokens, lengths).softmax(-1).cpu().numpy())
    return np.concatenate(results)


def fit_transformer(train, validation, config, seed):
    seed_everything(seed)
    model = CareTransformer(**architecture(config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    dataset = TensorDataset(torch.from_numpy(train["x"]), torch.from_numpy(train["lengths"]), torch.from_numpy(train["y"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, generator=generator)
    best_loss, best_epoch, best_state, stale, history = float("inf"), 0, None, 0, []
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for tokens, lengths, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, lengths)
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(target)
        scores = classification_metrics(validation["y"], predict(model, validation))
        history.append({"epoch": epoch, "train_nll": total_loss / len(dataset),
                        "validation_nll": scores["nll"], "validation_accuracy": scores["accuracy"]})
        LOG.info("Graine %s · époque %s · perte validation %.4f · exactitude %.1f %%", seed, epoch, scores["nll"], 100*scores["accuracy"])
        if scores["nll"] < best_loss - 1e-5:
            best_loss, best_epoch, best_state, stale = scores["nll"], epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"seed": seed, "best_epoch": best_epoch, "best_validation_nll": best_loss,
                   "history": history, "training_seconds": time.perf_counter() - started,
                   "parameters": sum(p.numel() for p in model.parameters())}


def run_experiment(root, data, manifest, config):
    root = Path(root)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    train, validation, test = data["train"], data["validation"], data["test"]
    baseline_models = {name: NGram(order).fit(train) for name, order in [("Frequence", 0), ("Markov_1", 1), ("Markov_2", 2)]}
    write_json(root / "models/baselines.json", {name: model.serialize() for name, model in baseline_models.items()})
    # Le test n'est évalué qu'après la fixation de tous les checkpoints et de la graine choisie.
    runs, checkpoints = [], []
    for seed in config["model_seeds"]:
        model, run = fit_transformer(train, validation, config, seed)
        checkpoint = {"state_dict": model.state_dict(), "architecture": architecture(config),
                      "seed": seed, "events": EVENTS, "data_sha256": manifest["events_sha256"],
                      "best_epoch": run["best_epoch"], "synthetic_only": True}
        torch.save(checkpoint, root / f"models/transformer_seed_{seed}.pt")
        checkpoints.append(checkpoint)
        runs.append(run)
        write_json(root / "reports/training_progress.json", runs)
    selected = min(range(len(runs)), key=lambda i: runs[i]["best_validation_nll"])
    torch.save(checkpoints[selected], root / "models/best.pt")
    baseline_validation = {name: classification_metrics(validation["y"], model.probabilities(validation))
                           for name, model in baseline_models.items()}
    chosen_baseline = min(baseline_validation, key=lambda name: baseline_validation[name]["nll"])
    predictions = {name: model.probabilities(test) for name, model in baseline_models.items()}
    for index, checkpoint in enumerate(checkpoints):
        model = CareTransformer(**checkpoint["architecture"])
        model.load_state_dict(checkpoint["state_dict"])
        p = predict(model, test)
        runs[index]["test"] = classification_metrics(test["y"], p)
        if index == selected:
            predictions["Transformer"] = p
    metrics = {name: classification_metrics(test["y"], p) for name, p in predictions.items()}
    comparison = paired_patient_bootstrap(test["y"], predictions["Transformer"], predictions[chosen_baseline],
                                         test["patients"], config["bootstrap_repeats"])
    # Comparaison diagnostique : même checkpoint, entrée privée de tout sauf les deux derniers événements.
    model = CareTransformer(**checkpoints[selected]["architecture"])
    model.load_state_dict(checkpoints[selected]["state_dict"])
    short = {key: value.copy() for key, value in test.items()}
    for index, length in enumerate(test["lengths"]):
        short["x"][index] = 0
        short["x"][index, :2] = test["x"][index, length-2:length]
    short["lengths"][:] = 2
    short_metrics = classification_metrics(test["y"], predict(model, short))
    report = {"synthetic_only": True, "config": config, "data": manifest, "metrics": metrics,
              "baseline_validation": baseline_validation, "baseline_selected_on_validation": chosen_baseline,
              "transformer_runs": runs, "selected_seed": runs[selected]["seed"],
              "selection_rule": "Époque et graine choisies par NLL de validation ; aucune sélection sur test",
              "paired_comparison": comparison, "short_context_diagnostic": short_metrics,
              "short_context_warning": "Troncature sans réentraînement et positions réinitialisées : changement de distribution, pas une ablation causale isolée.",
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                              "platform": platform.platform(), "device": "cpu", "threads": config["threads"]}}
    write_json(root / "reports/results.json", report)
    np.savez_compressed(root / "data/test_predictions.npz", y=test["y"], patients=test["patients"], **predictions)
    # Exemples déterministes : six patients test, une fenêtre de chacun ; pas de sélection sur succès.
    _, first = np.unique(test["patients"], return_index=True)
    examples = []
    for i in first[:6]:
        top = np.argsort(-predictions["Transformer"][i])[:3]
        examples.append({"patient": str(test["patients"][i]),
                         "history": [EVENTS[token-1] for token in test["x"][i, :test["lengths"][i]]],
                         "actual_next": EVENTS[test["y"][i]],
                         "top3": [{"event": EVENTS[j], "probability": float(predictions["Transformer"][i, j])} for j in top]})
    write_json(root / "reports/examples.json", examples)
    return report
