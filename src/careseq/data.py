"""Parcours fictifs puis fenêtres historiques ; séparation par patient avant fenêtrage."""
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

EVENTS = ["GENERALISTE", "BIOLOGIE", "CARDIOLOGIE", "PNEUMOLOGIE", "ORTHOPEDIE",
          "IMAGERIE", "PHARMACIE", "URGENCES", "HOSPITALISATION", "REEDUCATION",
          "SUIVI_CARDIO", "SUIVI_PNEUMO", "SUIVI_ORTHO", "RETOUR_DOMICILE"]
TOKEN_TO_ID = {name: index + 1 for index, name in enumerate(EVENTS)}
N_CLASSES = len(EVENTS)
PAD = 0


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(patients=3000, seed=2026):
    if patients < 20:
        raise ValueError("Au moins 20 patients pour une séparation exploitable")
    rng = np.random.default_rng(seed)
    records = []
    for patient in range(patients):
        # Profils et règles arbitraires, pas un modèle médical ni une calibration Open DAMIR.
        profile = int(rng.integers(0, 3))
        specialist = ["CARDIOLOGIE", "PNEUMOLOGIE", "ORTHOPEDIE"][profile]
        followup = ["SUIVI_CARDIO", "SUIVI_PNEUMO", "SUIVI_ORTHO"][profile]
        path = ["GENERALISTE", "BIOLOGIE", specialist]
        length = int(rng.integers(18, 37))
        while len(path) < length:
            if rng.random() < 0.15:
                motif = ["URGENCES", "IMAGERIE", "HOSPITALISATION", "RETOUR_DOMICILE", followup]
            elif profile == 2 and rng.random() < 0.30:
                motif = ["IMAGERIE", "REEDUCATION", "REEDUCATION", followup]
            else:
                # Les deux événements précédant le suivi ne révèlent pas toujours le profil.
                motif = ["PHARMACIE", "GENERALISTE", "BIOLOGIE", followup]
            for event in motif:
                if rng.random() < 0.10:
                    path.append(str(rng.choice(["GENERALISTE", "BIOLOGIE", "PHARMACIE", "IMAGERIE"])))
                path.append(event)
        path = path[:length]
        timestamp = date(2023, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
        for index, event in enumerate(path):
            timestamp += timedelta(days=int(rng.integers(1, 31)))
            records.append({"patient_id": f"SYN_{patient:05d}", "event_index": index,
                            "event_date": timestamp, "event": event,
                            "event_id": TOKEN_TO_ID[event], "synthetic": True})
    return pl.DataFrame(records)


def split_patients(ids, seed=31):
    ids = np.array(sorted(set(ids)))
    order = np.random.default_rng(seed).permutation(len(ids))
    train_end, val_end = int(0.70 * len(ids)), int(0.85 * len(ids))
    return {"train": ids[order[:train_end]].tolist(), "validation": ids[order[train_end:val_end]].tolist(),
            "test": ids[order[val_end:]].tolist()}


def windows(frame, ids, max_context=24, min_context=3):
    if not 1 <= min_context <= max_context:
        raise ValueError("Contexte invalide")
    xs, ys, lengths, patients, positions = [], [], [], [], []
    selected = frame.filter(pl.col("patient_id").is_in(ids)).sort(["patient_id", "event_index"])
    for patient in selected.partition_by("patient_id", maintain_order=True):
        events = patient["event_id"].to_list()
        identifier = patient["patient_id"][0]
        for position in range(min_context, len(events)):
            context = events[max(0, position - max_context):position]
            x = np.zeros(max_context, dtype=np.int64)
            x[:len(context)] = context
            xs.append(x)
            ys.append(events[position] - 1)  # Classes de sortie 0..13 ; PAD jamais une cible.
            lengths.append(len(context))
            patients.append(identifier)
            positions.append(position)
    if not xs:
        raise ValueError("Aucune fenêtre exploitable")
    return {"x": np.stack(xs), "y": np.array(ys, dtype=np.int64),
            "lengths": np.array(lengths, dtype=np.int64), "patients": np.array(patients),
            "target_position": np.array(positions, dtype=np.int64)}


def prepare(root, config):
    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    frame = generate(config["patients"], config["data_seed"])
    frame.write_parquet(root / "data/events.parquet", compression="zstd")
    splits = split_patients(frame["patient_id"].unique().to_list(), config["split_seed"])
    result = {split: windows(frame, ids, config["max_context"], config["min_context"])
              for split, ids in splits.items()}
    sets = {name: set(ids) for name, ids in splits.items()}
    disjoint = not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    if not disjoint:
        raise ValueError("Patients partagés entre partitions")
    support = {name: np.bincount(data["y"], minlength=N_CLASSES).tolist() for name, data in result.items()}
    if min(support["train"]) == 0:
        raise ValueError("Une classe n'a aucun exemple d'entraînement ; augmenter les données ou revoir le contexte minimal")
    write_json(root / "reports/data_quality.json", {"patient_splits_disjoint": disjoint,
                "all_training_classes_present": True, "target_support": support, "events": EVENTS})
    for split, data in result.items():
        np.savez_compressed(root / f"data/{split}.npz", **data)
    write_json(root / "data/splits.json", splits)
    write_json(root / "data/vocabulary.json", {"padding_id": PAD, "token_to_id": TOKEN_TO_ID,
                                               "target_class": {e: i for i, e in enumerate(EVENTS)}})
    metadata = {"synthetic": True, "patients": config["patients"], "events": frame.height,
                "data_seed": config["data_seed"], "split_seed": config["split_seed"],
                "events_sha256": file_hash(root / "data/events.parquet"),
                "splits": {s: {"patients": len(splits[s]), "examples": len(data["y"])} for s, data in result.items()},
                "features": "types des événements passés uniquement ; dates utilisées pour l'ordre, pas comme entrée du modèle",
                "limitations": "Règles inventées, pas de calibration médicale ; patients test nouveaux, même générateur."}
    write_json(root / "data/manifest.json", metadata)
    return result, metadata
