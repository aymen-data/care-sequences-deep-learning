import numpy as np

from .data import N_CLASSES


def classification_metrics(target, probabilities):
    if probabilities.shape != (len(target), N_CLASSES):
        raise ValueError("Dimensions des probabilités incorrectes")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Probabilités invalides")
    prediction = probabilities.argmax(axis=1)
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(confusion, (target, prediction), 1)
    tp = confusion.diagonal().astype(float)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    return {"accuracy": float((prediction == target).mean()),
            "top3_accuracy": float((np.argsort(-probabilities, axis=1, kind="stable")[:, :3] == target[:, None]).any(axis=1).mean()),
            "macro_f1": float(f1.mean()),
            "nll": float(-np.log(np.clip(probabilities[np.arange(len(target)), target], 1e-12, 1)).mean()),
            "confusion": confusion.tolist(), "f1_per_class": f1.tolist(),
            "recall_per_class": recall.tolist(), "support_per_class": support.tolist()}


def paired_patient_bootstrap(target, prob_a, prob_b, patients, repeats=1000, seed=123):
    """Les fenêtres d'un patient sont corrélées : rééchantillonnage de patients entiers."""
    ids, inverse = np.unique(patients, return_inverse=True)
    n = np.bincount(inverse)
    a = np.bincount(inverse, weights=(prob_a.argmax(1) == target))
    b = np.bincount(inverse, weights=(prob_b.argmax(1) == target))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        draw = rng.integers(len(ids), size=len(ids))
        differences.append(float((a[draw].sum() - b[draw].sum()) / n[draw].sum()))
    return {"delta_accuracy": float((a.sum()-b.sum()) / n.sum()),
            "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
            "patients": len(ids), "repeats": repeats, "seed": seed,
            "method": "bootstrap apparié par patient, pondération par nombre de prédictions",
            "limitation": "Intervalle conditionnel aux modèles entraînés et à ce générateur ; pas une validité clinique."}
