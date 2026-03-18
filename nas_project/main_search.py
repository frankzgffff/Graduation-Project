from __future__ import annotations

import argparse
import json
from pathlib import Path

from nas_project.config import build_default_config, load_config, set_seed
from nas_project.evolution.evolution import HybridRLEvolutionSearch
from nas_project.search.search_space import SearchSpace
from nas_project.trainer.evaluator import ArchitectureEvaluator
from nas_project.utils.logger import ExperimentLogger, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid RL-EA based Neural Architecture Search")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON experiment config.")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name, e.g. cifar10 or fake")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--epochs-per-eval", type=int, default=None)
    parser.add_argument("--disable-rl", action="store_true")
    parser.add_argument("--disable-surrogate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config is not None else build_default_config(smoke_test=args.smoke_test)
    if args.smoke_test:
        config = build_default_config(smoke_test=True)

    if args.exp_name is not None:
        config.exp_name = args.exp_name
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed

    if args.dataset is not None:
        config.dataset.name = args.dataset
    if args.generations is not None:
        config.search.generations = args.generations
    if args.population_size is not None:
        config.search.population_size = args.population_size
    if args.epochs_per_eval is not None:
        config.train.epochs_per_eval = args.epochs_per_eval
    if args.disable_rl:
        config.search.use_rl = False
    if args.disable_surrogate:
        config.search.use_surrogate = False

    set_seed(config.seed)
    run_dir = Path(config.output_dir) / config.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.json")

    logger = setup_logger(run_dir, name=config.exp_name)
    exp_logger = ExperimentLogger(run_dir)

    search_space = SearchSpace(
        num_blocks=config.model.num_blocks,
        op_names=config.search.op_names,
        depth_choices=config.model.depth_choices,
        width_choices=config.model.width_choices,
    )
    evaluator = ArchitectureEvaluator(config)
    searcher = HybridRLEvolutionSearch(
        config=config,
        search_space=search_space,
        evaluator=evaluator,
        logger=logger,
        exp_logger=exp_logger,
        output_dir=run_dir,
    )
    best = searcher.run()
    final_metrics = evaluator.retrain_best(best.architecture, run_dir)

    summary = {
        "best_uid": best.uid,
        "best_fitness": best.fitness,
        "search_metrics": best.metrics,
        "final_retrain_metrics": final_metrics,
        "architecture": best.architecture.to_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Best summary saved to %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
