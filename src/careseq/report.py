"""Figures scientifiques exportables et rapport HTML local, sans dépendance réseau."""
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import EVENTS

COLORS = ["#b8bccb", "#929bb8", "#66759b", "#5144ae"]


def figures(root, result):
    folder = root / "reports/figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    names = list(result["metrics"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for shift, metric, label, alpha in [(-0.18, "accuracy", "Exactitude", 1), (0.18, "macro_f1", "F1 macro", 0.55)]:
        values = [result["metrics"][name][metric] for name in names]
        bars = ax.bar(x + shift, values, 0.34, color=COLORS, alpha=alpha, label=label)
        ax.bar_label(bars, labels=[f"{100*v:.1f}%" for v in values], fontsize=8, padding=3)
    ax.set_xticks(x, ["Fréquence", "Dernier événement", "Deux événements", "Transformer"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score sur patients test")
    ax.set_title("Comparaison sur des parcours synthétiques")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(folder / "comparaison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for run in result["transformer_runs"]:
        history = run["history"]
        ax.plot([h["epoch"] for h in history], [h["validation_nll"] for h in history], marker="o", ms=4,
                label=f"Validation · graine {run['seed']}")
        if run["seed"] == result["selected_seed"]:
            ax.plot([h["epoch"] for h in history], [h["train_nll"] for h in history], "--", color="#444444", label="Train · graine retenue")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Perte logarithmique (NLL, plus bas = mieux)")
    ax.set_title("Apprentissage · données synthétiques")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(folder / "apprentissage.png", dpi=160)
    plt.close(fig)
    confusion = np.array(result["metrics"]["Transformer"]["confusion"])
    normalized = np.divide(confusion, confusion.sum(1, keepdims=True), out=np.zeros_like(confusion, dtype=float),
                           where=confusion.sum(1, keepdims=True) > 0)
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    heat = ax.imshow(normalized, vmin=0, vmax=1, cmap="Purples")
    ax.set_xticks(range(len(EVENTS)), EVENTS, rotation=90, fontsize=8)
    ax.set_yticks(range(len(EVENTS)), EVENTS, fontsize=8)
    for i in range(len(EVENTS)):
        for j in range(len(EVENTS)):
            if normalized[i, j] >= 0.04:
                ax.text(j, i, f"{normalized[i,j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if normalized[i,j] > 0.55 else "black")
    ax.set_xlabel("Événement prédit")
    ax.set_ylabel("Événement réellement généré")
    ax.set_title("Transformer retenu · matrice de confusion normalisée par ligne\nPatients test synthétiques")
    fig.colorbar(heat, ax=ax, shrink=0.75, label="Part des prédictions de la classe réelle")
    fig.savefig(folder / "confusion.png", dpi=160)
    plt.close(fig)


def render(root):
    root = Path(root)
    result = json.loads((root / "reports/results.json").read_text(encoding="utf-8"))
    examples = json.loads((root / "reports/examples.json").read_text(encoding="utf-8"))
    figures(root, result)
    metrics = result["metrics"]
    selected = next(r for r in result["transformer_runs"] if r["seed"] == result["selected_seed"])
    comparison = result["paired_comparison"]
    lower, upper = comparison["ci95"]
    if lower > 0:
        conclusion = "Le Transformer retenu dépasse la référence statistique retenue dans cette expérience synthétique."
    elif upper < 0:
        conclusion = "La référence statistique retenue obtient une meilleure exactitude dans cette expérience."
    else:
        conclusion = "Cette expérience ne permet pas de conclure clairement à un gain d’exactitude du Transformer sur la référence retenue."
    table_rows = "".join(f'<tr><td>{html.escape(name)}</td><td>{m["accuracy"]:.1%}</td><td>{m["top3_accuracy"]:.1%}</td><td>{m["macro_f1"]:.3f}</td><td>{m["nll"]:.3f}</td></tr>' for name, m in metrics.items())
    seed_rows = "".join(f'<tr><td>{r["seed"]}{" · retenue" if r["seed"] == result["selected_seed"] else ""}</td><td>{r["best_epoch"]}</td><td>{r["best_validation_nll"]:.3f}</td><td>{r["test"]["accuracy"]:.1%}</td><td>{r["training_seconds"]:.1f} s</td></tr>' for r in result["transformer_runs"])
    example_cards = "".join(f'<article><b>{html.escape(ex["patient"])}</b><p class="sequence">{" → ".join(map(html.escape, ex["history"]))}</p><p>Événement suivant généré : <strong>{html.escape(ex["actual_next"])}</strong></p><p>Top 3 : {" · ".join(html.escape(p["event"])+" "+format(p["probability"], ".0%") for p in ex["top3"])}</p></article>' for ex in examples[:3])
    doc = f'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Séquences de soins · Projet Deep Learning</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f3f8;color:#20213a;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:42px 28px}}header{{border-top:5px solid #5144ae;padding-top:22px}}.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#5144ae;font-weight:700}}h1{{font-size:clamp(32px,5vw,50px);line-height:1.12;letter-spacing:-.035em;margin:15px 0}}h2{{font-size:23px;margin:0 0 15px}}.intro{{font-size:18px;color:#66677c;max-width:850px}}.notice{{background:#fff3d5;border:1px solid #e8d3a0;border-radius:8px;padding:15px 18px;margin-top:20px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:25px 0}}.card,section{{background:white;border:1px solid #dedee9;border-radius:12px;padding:24px}}.card span{{font-size:12px;color:#66677c}}.card strong{{display:block;font-size:32px}}section{{margin:20px 0}}.flow{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}.flow b{{background:#eeecf9;border-radius:5px;padding:8px 12px;font-size:14px}}.note{{font-size:13px;color:#66677c}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}img{{width:100%;height:auto;display:block}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:12px 8px;border-bottom:1px solid #e4e4ed;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{color:#66677c;font-size:12px}}article{{border-left:3px solid #5144ae;background:#f8f7fc;padding:12px 18px;margin:14px 0;font-size:14px}}.sequence{{font-family:monospace;color:#5144ae}}a{{color:#5144ae}}footer{{font-size:13px;color:#66677c;margin:28px 0}}@media(max-width:700px){{main{{padding:24px 14px}}.cards,.two{{grid-template-columns:1fr}}section{{padding:18px;overflow:auto}}}}@media print{{body{{background:white}}section{{break-inside:avoid}}}}
</style></head><body><main><header><div class="eyebrow">Portfolio · Projet 02 · Deep Learning</div><h1>Lire une séquence.<br>Prédire l’événement suivant.</h1><p class="intro">Des historiques fictifs de soins transformés en embeddings, puis traités par un petit Transformer et comparés à des références statistiques.</p><div class="notice"><strong>Données entièrement synthétiques.</strong> Les scores mesurent l’apprentissage de règles inventées. Ils ne démontrent aucune performance clinique ni une reproduction des données individuelles du SNDS.</div></header>
<div class="cards"><div class="card"><span>PATIENTS FICTIFS</span><strong>{result["data"]["patients"]:,}</strong><span>70 % train · 15 % validation · 15 % test</span></div><div class="card"><span>EXACTITUDE DU TRANSFORMER</span><strong>{metrics["Transformer"]["accuracy"]:.1%}</strong><span>Sur {result["data"]["splits"]["test"]["examples"]:,} fenêtres de patients test</span></div><div class="card"><span>ÉCART À LA RÉFÉRENCE RETENUE</span><strong>{100*comparison["delta_accuracy"]:+.1f} points</strong><span>{html.escape(result["baseline_selected_on_validation"])} · sélection sur validation</span></div></div>
<section><h2>Du parcours au modèle</h2><div class="flow"><b>Événements passés</b><span>→</span><b>Identifiants</b><span>→</span><b>Embeddings + positions</b><span>→</span><b>Transformer</b><span>→</span><b>14 probabilités</b></div><p class="note">{selected["parameters"]:,} paramètres · {result["config"]["layers"]} couches · {result["config"]["heads"]} têtes · dimension {result["config"]["d_model"]} · contexte maximal de {result["config"]["max_context"]} événements. Les dates ordonnent les données ; elles ne sont pas des caractéristiques du modèle.</p></section>
<section><h2>Une comparaison qui commence par les méthodes simples</h2><table><thead><tr><th>Modèle</th><th>Exactitude</th><th>Top 3</th><th>F1 macro</th><th>NLL ↓</th></tr></thead><tbody>{table_rows}</tbody></table><p><strong>{conclusion}</strong></p><p class="note">Écart Transformer − référence : {100*comparison["delta_accuracy"]:+.2f} points ; intervalle bootstrap à 95 % [{100*lower:+.2f} ; {100*upper:+.2f}] points. Rééchantillonnage apparié de patients entiers, {comparison["repeats"]} répétitions. Cet intervalle est conditionnel aux modèles et au générateur.</p></section>
<section><h2>Apprentissage et résultats</h2><div class="two"><img src="figures/comparaison.png" alt="Comparaison des modèles sur les patients test synthétiques"><img src="figures/apprentissage.png" alt="Courbes de perte d'entraînement et de validation"></div><p class="note">L’époque et la graine sont retenues sur la perte de validation. Le jeu de test intervient après cette sélection. Trois initialisations donnent une première indication de variabilité, sans remplacer une étude multi-jeux de données.</p><table><thead><tr><th>Graine</th><th>Époque retenue</th><th>NLL validation</th><th>Exactitude test</th><th>Entraînement</th></tr></thead><tbody>{seed_rows}</tbody></table></section>
<section><h2>Quelles catégories sont confondues ?</h2><img src="figures/confusion.png" alt="Matrice de confusion du Transformer sur les quatorze classes"><p class="note">Chaque ligne correspond à une classe réelle. Les spécialités initiales sont tirées au hasard après le même préfixe : certaines cibles sont donc intrinsèquement difficiles à prédire à ce stade.</p></section>
<section><h2>Exemples non sélectionnés selon leur réussite</h2><p class="note">Première fenêtre des premiers patients test triés. Avec seulement deux événements d’historique, la spécialité suivante n’est pas identifiable dans le générateur. Six exemples sont enregistrés dans examples.json.</p>{example_cards}</section>
<section><h2>Ce que démontre ce projet</h2><ul><li>Construire des fenêtres historiques sans donner la cible ou le futur au modèle.</li><li>Séparer les patients avant la création des fenêtres et apprendre les références uniquement sur train.</li><li>Entraîner, sauvegarder et recharger un Transformer avec PyTorch.</li><li>Évaluer plusieurs métriques et la variabilité des initialisations.</li></ul><p>Le générateur contient volontairement des motifs et une dépendance à l’historique. La comparaison ne prouve pas la supériorité générale des Transformers. Les patients test proviennent du même générateur ; aucune validation temporelle, externe ou médicale n’est réalisée.</p><p class="note">Diagnostic complémentaire : en conservant seulement deux événements à l’entrée du même checkpoint, l’exactitude atteint {result["short_context_diagnostic"]["accuracy"]:.1%}. Cette troncature modifie la distribution et les positions ; elle n’isole pas causalement l’effet de la longueur du contexte.</p><p><a href="../README.md">Installation et utilisation</a> · <a href="../README.md">Comprendre et présenter le travail</a> · <a href="results.json">Résultats détaillés</a></p></section><footer>Projet pédagogique indépendant · données fictives · aucune recommandation de soins. Code et protocole documentés pour une reprise dans un environnement Python local.</footer></main></body></html>'''
    (root / "reports/rapport.html").write_text(doc, encoding="utf-8")
    summary = f"# Résultats — séquences synthétiques\n\n{result['data']['patients']} patients fictifs, {result['data']['events']} événements.\n\n{conclusion}\n\n| Modèle | Exactitude | Top 3 | F1 macro | NLL |\n|---|---:|---:|---:|---:|\n"
    summary += "\n".join(f"| {name} | {m['accuracy']:.2%} | {m['top3_accuracy']:.2%} | {m['macro_f1']:.3f} | {m['nll']:.3f} |" for name, m in metrics.items())
    summary += f"\n\nGraine retenue sur validation : {result['selected_seed']}. Écart d'exactitude à {result['baseline_selected_on_validation']} : {100*comparison['delta_accuracy']:+.2f} points, intervalle bootstrap par patient [{100*lower:+.2f} ; {100*upper:+.2f}].\n\nAucune validité clinique. Règles synthétiques inventées, test issu du même générateur.\n"
    (root / "reports/RESULTATS.md").write_text(summary, encoding="utf-8")
    with (root / "reports/metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "accuracy", "top3_accuracy", "macro_f1", "nll"])
        for name, m in metrics.items():
            writer.writerow([name, *[m[key] for key in ["accuracy", "top3_accuracy", "macro_f1", "nll"]]])
