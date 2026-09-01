#!/usr/bin/env python3
"""Orchestrate DSRL budget search, online comparison, and held-out evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VLM_DSRL_METHODS = {"guarded_joint", "joint", "dsrl", "vlm", "base", "flow_rwr"}
RL_METHODS = {"guarded_joint", "joint", "dsrl", "residual_rl", "flow_rwr"}
DEFAULT_METHODS = (
    "guarded_joint",
    "joint",
    "dsrl",
    "vlm",
    "residual_rl",
    "flow_rwr",
    "base",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "matrix", "all", "analyze"), default="all")
    parser.add_argument(
        "--bc-policy",
        type=Path,
        default=Path("outputs/lab_pick_dinov3_flow_bc200_yaw0/best.pt"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("exp_report/compare_exp"))
    parser.add_argument("--search-budgets", type=int, nargs="+", default=(100, 150, 200))
    parser.add_argument("--train-interactions", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--train-seed", type=int, default=5200)
    parser.add_argument("--search-eval-seed", type=int, default=24200)
    parser.add_argument("--eval-seed", type=int, default=25200)
    parser.add_argument("--break-force-threshold-n", type=float, default=4.5)
    parser.add_argument("--actor-lr", type=float, default=3.0e-5)
    parser.add_argument("--critic-lr", type=float, default=3.0e-5)
    parser.add_argument("--alpha-lr", type=float, default=3.0e-5)
    parser.add_argument("--initial-log-std", type=float, default=-2.0)
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--advisor", choices=("deterministic", "openai"), default="deterministic")
    parser.add_argument(
        "--evaluation-policy",
        choices=("deterministic", "stochastic"),
        default="stochastic",
        help="Frozen DSRL-bearing SAC policies are sampled once per seeded decision by default.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_tag(value: float) -> str:
    """Return a filesystem-safe scientific-notation tag."""

    return f"{float(value):.0e}".replace("+", "p").replace("-", "m")


def _complete(
    path: Path,
    *,
    episodes: int | None = None,
    interactions: int | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    if episodes is not None and int(payload.get("completed_episodes", -1)) != int(episodes):
        return False
    if interactions is not None and int(payload.get("outer_interactions", -1)) != int(interactions):
        return False
    return episodes is not None or interactions is not None


class Runner:
    def __init__(self, args: argparse.Namespace, repo_root: Path) -> None:
        self.args = args
        self.repo_root = repo_root
        self.output_root = (repo_root / args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.console_dir = self.output_root / "console_logs"
        self.console_dir.mkdir(exist_ok=True)
        self.bc_policy = (repo_root / args.bc_policy).resolve()
        if not self.bc_policy.is_file():
            raise FileNotFoundError(self.bc_policy)
        self.vlm_launcher = repo_root / "scripts/reinforcement_learning/vlm_dsrl/run_experiment.py"
        self.residual_launcher = Path(__file__).with_name("run_residual_experiment.py")

    def _run(
        self,
        name: str,
        command: list[str],
        result: Path,
        *,
        episodes: int | None = None,
        interactions: int | None = None,
    ) -> Path:
        if self.args.resume and _complete(
            result, episodes=episodes, interactions=interactions
        ):
            print(f"[SKIP] {name}: complete", flush=True)
            return result
        run_dir = result.parent
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                f"Incomplete non-empty run requires inspection or a new output root: {run_dir}"
            )
        print("[RUN]", " ".join(command), flush=True)
        if self.args.dry_run:
            return result
        env = os.environ.copy()
        env.setdefault(
            "TACEX_ISAAC_PYTHON",
            "/home/limx/anaconda3/envs/env_isaaclab/bin/python",
        )
        log_path = self.console_dir / f"{name}.log"
        with log_path.open("x", encoding="utf-8", buffering=1) as stream:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                if line.startswith(("[EPISODE]", "[EVALUATION]", "[SUMMARY]")):
                    print(f"[{name}] {line}", end="", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")
        if not _complete(result, episodes=episodes, interactions=interactions):
            expected = (
                f"{episodes} completed episodes"
                if episodes is not None
                else f"{interactions} outer interactions"
            )
            raise RuntimeError(f"{name} did not write {expected}")
        return result

    def _common_vlm_command(
        self,
        *,
        method: str,
        phase: str,
        output_dir: Path,
        episodes: int,
        seed: int,
        bc_policy: Path | None = None,
        dsrl_checkpoint: Path | None = None,
        training_interactions: int | None = None,
    ) -> list[str]:
        command = [
            sys.executable,
            str(self.vlm_launcher),
            "--mode",
            method,
            "--phase",
            phase,
            "--bc-policy",
            str(self.bc_policy if bc_policy is None else bc_policy),
            "--output-dir",
            str(output_dir),
            "--num-episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--break-force-threshold-n",
            str(self.args.break_force_threshold_n),
            "--labware-random-yaw-deg",
            "0",
            "--physical-force-range-n",
            "0.5",
            "3.8",
            "--initial-force-range-n",
            "0.8",
            "1.4",
            "--advisor",
            self.args.advisor,
            "--evaluation-policy",
            self.args.evaluation_policy,
            "--actor-lr",
            str(self.args.actor_lr),
            "--critic-lr",
            str(self.args.critic_lr),
            "--alpha-lr",
            str(self.args.alpha_lr),
            "--initial-log-std",
            str(self.args.initial_log_std),
            "--learning-starts",
            "32",
            "--batch-size",
            "32",
        ]
        if dsrl_checkpoint is not None:
            command.extend(("--dsrl-checkpoint", str(dsrl_checkpoint)))
        if training_interactions is not None:
            command.extend(("--training-interactions", str(training_interactions)))
        return command

    def run_vlm_method(
        self,
        *,
        method: str,
        phase: str,
        output_dir: Path,
        episodes: int,
        seed: int,
        bc_policy: Path | None = None,
        dsrl_checkpoint: Path | None = None,
        training_interactions: int | None = None,
    ) -> Path:
        name = output_dir.name
        result = output_dir / "results.json"
        return self._run(
            name,
            self._common_vlm_command(
                method=method,
                phase=phase,
                output_dir=output_dir,
                episodes=episodes,
                seed=seed,
                bc_policy=bc_policy,
                dsrl_checkpoint=dsrl_checkpoint,
                training_interactions=training_interactions,
            ),
            result,
            episodes=None if training_interactions is not None else episodes,
            interactions=training_interactions,
        )

    def run_residual(
        self,
        *,
        phase: str,
        output_dir: Path,
        episodes: int,
        seed: int,
        checkpoint: Path | None = None,
        training_interactions: int | None = None,
    ) -> Path:
        command = [
            sys.executable,
            str(self.residual_launcher),
            "--phase",
            phase,
            "--bc-policy",
            str(self.bc_policy),
            "--output-dir",
            str(output_dir),
            "--num-episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--break-force-threshold-n",
            str(self.args.break_force_threshold_n),
            "--labware-random-yaw-deg",
            "0",
            "--learning-starts",
            "256",
            "--batch-size",
            "256",
        ]
        if checkpoint is not None:
            command.extend(("--checkpoint", str(checkpoint)))
        if training_interactions is not None:
            command.extend(("--training-interactions", str(training_interactions)))
        return self._run(
            output_dir.name,
            command,
            output_dir / "results.json",
            episodes=None if training_interactions is not None else episodes,
            interactions=training_interactions,
        )

    def search(self) -> dict[str, Any]:
        search_root = self.output_root / "budget_search"
        base_result = self.run_vlm_method(
            method="base",
            phase="evaluation",
            output_dir=search_root / f"base_eval_seed{self.args.search_eval_seed}_n{self.args.eval_episodes}",
            episodes=self.args.eval_episodes,
            seed=self.args.search_eval_seed,
        )
        if self.args.dry_run:
            return {"selected_budget": None, "runs": []}
        base = _read_json(base_result)
        trials = []
        selected = None
        actor_lr_tag = _float_tag(self.args.actor_lr)
        for budget in sorted(set(self.args.search_budgets)):
            train_dir = search_root / (
                f"dsrl_train_seed{self.args.train_seed}_i{budget}_alr{actor_lr_tag}"
            )
            train_result = self.run_vlm_method(
                method="dsrl",
                phase="online_training",
                output_dir=train_dir,
                episodes=budget,
                seed=self.args.train_seed,
                training_interactions=budget,
            )
            checkpoint = Path(_read_json(train_result)["learned_checkpoint"])
            eval_dir = search_root / (
                f"dsrl_eval_{self.args.evaluation_policy}_train_i{budget}_"
                f"alr{actor_lr_tag}_seed{self.args.search_eval_seed}_n{self.args.eval_episodes}"
            )
            eval_result = self.run_vlm_method(
                method="dsrl",
                phase="evaluation",
                output_dir=eval_dir,
                episodes=self.args.eval_episodes,
                seed=self.args.search_eval_seed,
                dsrl_checkpoint=checkpoint,
            )
            evaluated = _read_json(eval_result)
            exceeds = int(evaluated["successes"]) > int(base["successes"])
            trials.append(
                {
                    "train_interactions": budget,
                    "train_result": str(train_result),
                    "checkpoint": str(checkpoint),
                    "evaluation_result": str(eval_result),
                    "successes": evaluated["successes"],
                    "success_rate": evaluated["success_rate"],
                    "strictly_exceeds_base": exceeds,
                    "actor_lr": self.args.actor_lr,
                    "critic_lr": self.args.critic_lr,
                    "alpha_lr": self.args.alpha_lr,
                    "initial_log_std": self.args.initial_log_std,
                }
            )
            if exceeds:
                selected = budget
                break
        payload = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "definition": (
                "Train for the stated number of outer DSRL decisions, freeze the "
                "policy, then evaluate on 50 held-out explicitly paired seeds."
            ),
            "base_result": str(base_result),
            "base_successes": base["successes"],
            "base_success_rate": base["success_rate"],
            "evaluation_seed_start": self.args.search_eval_seed,
            "evaluation_policy": self.args.evaluation_policy,
            "actor_lr": self.args.actor_lr,
            "critic_lr": self.args.critic_lr,
            "alpha_lr": self.args.alpha_lr,
            "initial_log_std": self.args.initial_log_std,
            "selected_budget": selected,
            "runs": trials,
        }
        (search_root / "budget_search.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[SEARCH] selected_budget={selected}", flush=True)
        return payload

    def _selected_budget(self) -> int:
        if self.args.train_interactions is not None:
            return self.args.train_interactions
        if self.args.dry_run:
            return int(sorted(set(self.args.search_budgets))[0])
        path = self.output_root / "budget_search/budget_search.json"
        if not path.is_file():
            raise RuntimeError("Run --stage search first or pass --train-interactions.")
        selected = _read_json(path).get("selected_budget")
        if selected is None:
            raise RuntimeError("No tested DSRL budget strictly exceeded the base policy.")
        return int(selected)

    def matrix(self) -> dict[str, Any]:
        budget = self._selected_budget()
        online_root = self.output_root / "comparison" / "online"
        eval_root = self.output_root / "comparison" / "evaluation"
        entries: list[dict[str, Any]] = []
        search_path = self.output_root / "budget_search/budget_search.json"
        search_trials = [] if not search_path.is_file() else _read_json(search_path).get("runs", [])
        for method in self.args.methods:
            reused_search = next(
                (
                    item
                    for item in search_trials
                    if method == "dsrl"
                    and int(item["train_interactions"]) == budget
                    and bool(
                        item.get("selected", item.get("strictly_exceeds_base", False))
                    )
                ),
                None,
            )
            method_interactions = budget * 32 if method == "residual_rl" else budget
            train_dir = online_root / (
                f"{method}_seed{self.args.train_seed}_i{method_interactions}"
            )
            if reused_search is not None:
                train_result = Path(reused_search["train_result"])
            elif method == "residual_rl":
                train_result = self.run_residual(
                    phase="online_training",
                    output_dir=train_dir,
                    episodes=budget,
                    seed=self.args.train_seed,
                    training_interactions=method_interactions,
                )
            else:
                train_result = self.run_vlm_method(
                    method=method,
                    phase="online_training",
                    output_dir=train_dir,
                    episodes=budget,
                    seed=self.args.train_seed,
                    training_interactions=method_interactions,
                )
            if self.args.dry_run:
                entries.append({"method": method, "online_result": str(train_result)})
                continue
            train_payload = _read_json(train_result)
            eval_dir = eval_root / (
                f"{method}_{self.args.evaluation_policy}_train_i{method_interactions}_"
                f"seed{self.args.eval_seed}_n{self.args.eval_episodes}"
            )
            if method == "residual_rl":
                eval_result = self.run_residual(
                    phase="evaluation",
                    output_dir=eval_dir,
                    episodes=self.args.eval_episodes,
                    seed=self.args.eval_seed,
                    checkpoint=Path(train_payload["learned_checkpoint"]),
                )
            elif method == "flow_rwr":
                eval_result = self.run_vlm_method(
                    method="flow_rwr",
                    phase="evaluation",
                    output_dir=eval_dir,
                    episodes=self.args.eval_episodes,
                    seed=self.args.eval_seed,
                    bc_policy=Path(train_payload["learned_checkpoint"]),
                )
            else:
                dsrl_checkpoint = (
                    Path(train_payload["learned_checkpoint"])
                    if method in {"guarded_joint", "joint", "dsrl"}
                    else None
                )
                eval_result = self.run_vlm_method(
                    method=method,
                    phase="evaluation",
                    output_dir=eval_dir,
                    episodes=self.args.eval_episodes,
                    seed=self.args.eval_seed,
                    dsrl_checkpoint=dsrl_checkpoint,
                )
            entries.append(
                {
                    "method": method,
                    "training_outer_interactions": method_interactions,
                    "online_result": str(train_result),
                    "evaluation_result": str(eval_result),
                    "reused_training_from_budget_search": reused_search is not None,
                }
            )
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dsrl_outer_interaction_budget": budget,
            "nominal_training_physics_steps": budget * 32 * 2,
            "residual_outer_interaction_budget": budget * 32,
            "evaluation_episodes": self.args.eval_episodes,
            "train_seed": self.args.train_seed,
            "evaluation_seed_start": self.args.eval_seed,
            "dsrl_evaluation_policy": self.args.evaluation_policy,
            "break_force_threshold_n": self.args.break_force_threshold_n,
            "dsrl_actor_lr": self.args.actor_lr,
            "dsrl_critic_lr": self.args.critic_lr,
            "dsrl_alpha_lr": self.args.alpha_lr,
            "dsrl_initial_log_std": self.args.initial_log_std,
            "bc_policy": str(self.bc_policy),
            "methods": entries,
        }
        target = self.output_root / "comparison/manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest


def main() -> None:
    args = parse_args()
    if min(args.search_budgets) < 1 or args.eval_episodes < 1:
        raise ValueError("Interaction and evaluation budgets must be positive.")
    repo_root = Path(__file__).resolve().parents[3]
    runner = Runner(args, repo_root)
    if args.stage in {"search", "all"}:
        search = runner.search()
        if not args.dry_run and search["selected_budget"] is None:
            raise RuntimeError(
                "No search budget exceeded base; extend --search-budgets before the matrix."
            )
    if args.stage in {"matrix", "all"}:
        runner.matrix()
    if args.stage in {"analyze", "matrix", "all"} and not args.dry_run:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("analyze_results.py")),
                "--root",
                str(runner.output_root),
            ],
            cwd=repo_root,
            check=True,
        )


if __name__ == "__main__":
    main()
