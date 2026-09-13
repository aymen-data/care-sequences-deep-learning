import argparse
import json
import logging
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Projet pédagogique de séquences SYNTHÉTIQUES de soins")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path)
    sub.add_parser("report")
    infer = sub.add_parser("predict")
    infer.add_argument("--events", nargs="+", required=True)
    args = parser.parse_args()
    logs = args.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(logs / "experiment.log", encoding="utf-8")])
    if args.command == "run":
        from .data import prepare, write_json
        from .train import run_experiment
        config = json.loads((args.config or args.root / "config.json").read_text(encoding="utf-8"))
        write_json(args.root / "reports/config_used.json", config)
        logging.info("Génération de %s patients fictifs", config["patients"])
        data, manifest = prepare(args.root, config)
        logging.info("Fenêtres : %s", manifest["splits"])
        run_experiment(args.root, data, manifest, config)
        from .report import render
        render(args.root)
    elif args.command == "report":
        from .report import render
        render(args.root)
    else:
        from .inference import infer_events
        print(json.dumps(infer_events(args.root, args.events), ensure_ascii=False, indent=2))
    logging.info("Terminé : %s", args.command)


if __name__ == "__main__":
    main()
