from pathlib import Path

import torch

from .data import EVENTS, TOKEN_TO_ID
from .model import CareTransformer


def infer_events(root, events):
    unknown = [e for e in events if e not in TOKEN_TO_ID]
    if not events or unknown:
        raise ValueError(f"Événements attendus parmi {EVENTS}. Inconnus : {unknown}")
    checkpoint = torch.load(Path(root) / "models/best.pt", map_location="cpu", weights_only=True)
    model = CareTransformer(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.set_num_threads(4)
    context = events[-model.max_context:]
    tokens = torch.tensor([[TOKEN_TO_ID[e] for e in context]])
    with torch.inference_mode():
        probabilities = model(tokens, torch.tensor([len(context)])).softmax(-1)[0]
    order = probabilities.argsort(descending=True)[:3]
    return {"synthetic_only": True, "warning": "Démonstration du générateur fictif, aucune recommandation de soins.",
            "history_used": context, "context_truncated": len(context) < len(events),
            "top3": [{"event": EVENTS[i], "probability": float(probabilities[i])} for i in order.tolist()]}
