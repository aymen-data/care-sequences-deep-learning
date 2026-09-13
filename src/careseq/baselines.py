from collections import defaultdict

import numpy as np

from .data import N_CLASSES


class NGram:
    """Comptages appris sur train ; repli sur un contexte plus court si jamais observé."""
    def __init__(self, order=0, alpha=0.5):
        if order not in [0, 1, 2] or alpha <= 0:
            raise ValueError("Ordre 0, 1 ou 2 et lissage positif requis")
        self.order, self.alpha = order, alpha
        self.counts = [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in range(order + 1)]

    def fit(self, data):
        for x, length, target in zip(data["x"], data["lengths"], data["y"]):
            for order in range(self.order + 1):
                key = tuple(x[length-order:length]) if order else ()
                self.counts[order][key][target] += 1
        return self

    def probabilities(self, data):
        result = []
        for x, length in zip(data["x"], data["lengths"]):
            for order in range(min(self.order, length), -1, -1):
                key = tuple(x[length-order:length]) if order else ()
                if key in self.counts[order]:
                    count = self.counts[order][key]
                    result.append((count + self.alpha) / (count.sum() + self.alpha * N_CLASSES))
                    break
            else:
                raise ValueError("Référence non entraînée")
        return np.stack(result)

    def serialize(self):
        return {"order": self.order, "alpha": self.alpha,
                "counts": [{",".join(map(str, key)): value.tolist() for key, value in table.items()}
                           for table in self.counts]}
