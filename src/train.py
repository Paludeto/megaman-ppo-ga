"""
Module: src.train
Description: PPO training to defeat a Mega Man 1 boss from pixels.

The recipe (hyperparameters + reward function) is THE SAME for every boss and is fixed in the
constants below; only the learning rate and entropy coefficient can be overridden, for the tuning.
Early stop: by default, training stops at the first validation episode that ends with a no-damage
win (boss_hp == 0 and hp == 28, read from info); --no-early-stop trains to --timesteps.
--train-sticky P trains with per-frame sticky actions (Machado et al. 2018) in the TRAINING envs
only; validation stays on the deterministic evaluation task.
"""

import os
import re
import sys
import json
import time
import argparse
import platform
import subprocess
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.logger import TensorBoardOutputFormat

try:
    from src.env import make_venv
    from src.eval import INITIAL_HP, is_no_damage, is_win
except ImportError:
    from env import make_venv
    from eval import INITIAL_HP, is_no_damage, is_win

# ----------------------------------------------------------------------------------------------
# STANDARD RECIPE (same for every boss).
# ----------------------------------------------------------------------------------------------
D = 0.05                      # reward scale: +d per hit on the boss, -d per damage taken
DAMAGE_PENALTY_MULT = 2.0     # TRAINING penalizes damage 2x (forces dodging); evaluation uses 1.0 (real)
WIN_BONUS = 0.5               # terminal win bonus during TRAINING
FRAMESKIP = 4                 # frames per action (training and evaluation)
NORM_REWARD = True            # normalize the reward only (not the observation) — eval-safe

# PPO hyperparameters
LR = 2.5e-4                   # initial learning rate (decays linearly to 0)
ENT_COEF = 5e-3
TARGET_KL = 0.05
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
VF_COEF = 1.0
MAX_GRAD_NORM = 0.5
N_STEPS = 1024
BATCH_SIZE = 128
N_EPOCHS = 4
N_EVAL_EPISODES = 1
SEED = 666


def lr_at_step(step: float, initial_value: float, horizon: float) -> float:
    """
    Learning rate at absolute training step `step`: linear decay from `initial_value` (step 0) to 0
    at `horizon`, then constant at 0 (clamped) if training runs past the horizon.
    """
    return initial_value * max(0.0, 1.0 - step / horizon)


def linear_schedule(initial_value: float, horizon: int, total_steps: int):
    """
    Linear learning-rate schedule anchored to the ABSOLUTE step (`--lr-horizon`), not to SB3's
    progress_remaining. SB3 calls the function with progress_remaining = 1 - num_timesteps /
    total_timesteps; `total_steps` must be the value handed to learn(), which recovers the absolute
    step. With horizon == total_steps this equals progress_remaining * initial_value.
    """
    def func(progress_remaining: float) -> float:
        step = round((1.0 - progress_remaining) * total_steps)
        return lr_at_step(step, initial_value, horizon)
    return func


def vecnormalize_path_for(model_path: str) -> str:
    """
    Where the VecNormalize statistics matching a saved model live. CheckpointCallback writes
    `<prefix>_<n>_steps.zip` next to `<prefix>_vecnormalize_<n>_steps.pkl`; any other model
    (e.g. `final_model.zip`) uses `<stem>_vecnormalize.pkl`.
    """
    directory, name = os.path.split(model_path)
    match = re.fullmatch(r"(?P<prefix>.+)_(?P<steps>\d+)_steps(?:\.zip)?", name)
    if match:
        return os.path.join(directory, f"{match['prefix']}_vecnormalize_{match['steps']}_steps.pkl")
    stem = name[:-4] if name.endswith(".zip") else name
    return os.path.join(directory, f"{stem}_vecnormalize.pkl")


def _git_info(repo_dir: str) -> dict:
    """Commit hash and whether tracked files are modified (untracked files are ignored)."""
    def git(*cmd):
        return subprocess.run(["git", "-C", repo_dir, *cmd], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def collect_run_meta(args, device: str, repo_dir: str) -> dict:
    """Everything needed to identify a run: library versions, commit, hardware, CLI args and recipe."""
    import importlib.metadata as md
    import numpy
    import gymnasium
    import stable_baselines3
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "stable_retro": md.version("stable-retro"),
            "gymnasium": gymnasium.__version__,
            "numpy": numpy.__version__,
        },
        "git": _git_info(repo_dir),
        "hardware": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "cuda": torch.version.cuda if device == "cuda" else None,
            "cudnn": torch.backends.cudnn.version() if device == "cuda" else None,
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
        },
        # State AFTER model construction (SB3's seeding also touches the cuDNN flags).
        "torch_determinism": {
            "deterministic_torch_flag": args.deterministic_torch,
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "args": vars(args),
        "effective": {"lr_horizon": args.lr_horizon if args.lr_horizon is not None else args.timesteps,
                      "early_stop": "none" if getattr(args, "no_early_stop", False) else "no_damage_win"},
        "recipe": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, bool, str))},
    }


def write_run_meta(meta: dict, logs_dir: str) -> str:
    """Write `run_meta.json` in the log dir."""
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, "run_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


class RunMetaCallback(BaseCallback):
    """Writes the run metadata to TensorBoard (as text) when training starts."""

    def __init__(self, meta: dict):
        super().__init__()
        self.meta = meta

    def _on_training_start(self) -> None:
        text = "```json\n" + json.dumps(self.meta, indent=2, default=str) + "\n```"
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                fmt.writer.add_text("run_meta", text, self.num_timesteps)
                fmt.writer.flush()

    def _on_step(self) -> bool:
        return True


class ValidationCallback(EvalCallback):
    """
    EvalCallback that also classifies every validation episode from the final `info["episode"]`
    (VecMonitor records the final `hp` and `boss_hp`): win = boss_hp == 0 with hp > 0, no-damage win
    = win with hp == INITIAL_HP. It logs eval/win_rate and eval/no_damage_rate.

    With `stop_on_no_damage`, training stops after an evaluation whose episodes are ALL no-damage
    wins. Otherwise it behaves exactly like EvalCallback (the best_model is always chosen by
    mean reward).
    """

    def __init__(self, *args, stop_on_no_damage: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_on_no_damage = stop_on_no_damage
        self.last_outcomes = []  # one {"win", "no_damage"} per episode of the latest evaluation

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            info = locals_["info"]
            final = info.get("episode", info)  # VecMonitor's episode dict; the step info as fallback
            self.last_outcomes.append({"win": is_win(final["boss_hp"], final["hp"]),
                                       "no_damage": is_no_damage(final["boss_hp"], final["hp"])})
            for key, name in (("win", "eval/win_rate"), ("no_damage", "eval/no_damage_rate")):
                self.logger.record(name, sum(o[key] for o in self.last_outcomes) / len(self.last_outcomes))

    def _on_step(self) -> bool:
        evaluating = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluating:
            self.last_outcomes = []
        continue_training = super()._on_step()
        if evaluating and self.stop_on_no_damage and self.last_outcomes \
                and all(o["no_damage"] for o in self.last_outcomes):
            if self.verbose >= 1:
                print(f"Stopping training: no-damage win in validation at num_timesteps={self.num_timesteps}")
            return False
        return continue_training


def main():
    """
    Build the train/eval envs with the standard recipe, set up the checkpoint/eval callbacks with
    the no-damage stop and run training.
    """
    parser = argparse.ArgumentParser(description="Train PPO to defeat a boss (standard recipe; see the README).")
    parser.add_argument('--tag', type=str, default='run', help='Experiment name; namespaces logs/checkpoints/models')
    parser.add_argument('--state', type=str, required=True, help='Fight save state (e.g. Bombman-boss)')
    parser.add_argument('--episode-frames', type=int, default=7200, help='Frames per episode')
    parser.add_argument('--timesteps', type=int, default=40_000_000, help='Timestep ceiling; early-stop cuts it short on a win')
    parser.add_argument('--n-envs', type=int, default=16, help='Parallel environments')
    parser.add_argument('--seed', type=int, default=SEED, help='Seed for PPO/torch/numpy and the envs')
    parser.add_argument('--lr-horizon', type=int, default=None,
                        help='Absolute step at which the linear LR schedule reaches 0 (constant 0 afterwards). '
                             'Default: --timesteps')
    parser.add_argument('--deterministic-torch', action='store_true',
                        help='Deterministic torch/cuDNN/cuBLAS kernels (slower). Default keeps the current behavior')
    parser.add_argument('--train-sticky', type=float, default=0.0,
                        help='Per-frame sticky-action probability in the TRAINING envs (Machado et al. 2018; 0.25). '
                             'Validation stays deterministic. Default 0 = deterministic training')
    parser.add_argument('--lr', type=float, default=LR, help=f'Initial learning rate, decays linearly (default {LR})')
    parser.add_argument('--ent-coef', type=float, default=ENT_COEF, help=f'Entropy coefficient (default {ENT_COEF})')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Train to --timesteps, without the no-damage stop (best_model is still saved)')
    args = parser.parse_args()
    if not 0.0 <= args.train_sticky <= 1.0:
        parser.error("--train-sticky must be in [0, 1]")
    lr_horizon = args.lr_horizon if args.lr_horizon is not None else args.timesteps

    if args.deterministic_torch:
        # cuBLAS reads this when its handle is created, so it must be set before any CUDA use.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.cuda.is_available():
        # cuDNN optimization for fixed-shape inputs
        torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve paths relative to the project root, namespaced by tag
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    checkpoints_dir = os.path.join(project_root, "checkpoints", args.tag)
    models_dir = os.path.join(project_root, "models", f"{args.tag}_best")
    logs_dir = os.path.join(project_root, "logs", args.tag)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # TRAINING env: 2x damage penalty and win bonus.
    env = make_venv(n_envs=args.n_envs, state=args.state, d=D,
                    damage_penalty_mult=DAMAGE_PENALTY_MULT, win_bonus=WIN_BONUS,
                    frameskip=FRAMESKIP, norm_reward=NORM_REWARD,
                    max_episode_frames=args.episode_frames, sticky=args.train_sticky)
    # EVAL env: the real task (1x penalty, no win bonus), same frameskip/state/episode length, so
    # best_model and the early stop reflect a real win.
    eval_env = make_venv(n_envs=1, state=args.state, force_subproc=True, d=D,
                         frameskip=FRAMESKIP, norm_reward=NORM_REWARD,
                         max_episode_frames=args.episode_frames)
    if NORM_REWARD:
        eval_env.training = False
        eval_env.norm_reward = False

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 500000 // args.n_envs),
        save_path=checkpoints_dir,
        name_prefix=args.tag,
        save_vecnormalize=NORM_REWARD,  # <tag>_vecnormalize_<n>_steps.pkl next to each checkpoint
    )
    eval_callback = ValidationCallback(
        eval_env,
        best_model_save_path=models_dir,
        eval_freq=max(1, 100000 // args.n_envs),
        n_eval_episodes=N_EVAL_EPISODES,
        stop_on_no_damage=not args.no_early_stop,
        verbose=1,
    )

    learning_rate = linear_schedule(args.lr, lr_horizon, args.timesteps)

    print(f"Starting training from scratch (tag={args.tag})...")
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=args.ent_coef,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=TARGET_KL,
        policy_kwargs={"share_features_extractor": False},
        seed=args.seed,
        device=device,
        tensorboard_log=logs_dir,
    )

    meta = collect_run_meta(args, device, project_root)
    meta_path = write_run_meta(meta, logs_dir)
    print(f"Run metadata: {meta_path} (commit={meta['git']['commit']}, dirty={meta['git']['dirty']})")

    stop_rule = ("none (train to --timesteps)" if args.no_early_stop
                 else f"no-damage win (boss_hp == 0 and hp == {INITIAL_HP})")
    print(f"Training {args.tag}: state={args.state} early-stop: {stop_rule}, up to {args.timesteps} steps, "
          f"seed={args.seed}, lr_horizon={lr_horizon}, train_sticky={args.train_sticky}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[RunMetaCallback(meta), checkpoint_callback, eval_callback],
        tb_log_name="PPO",
        reset_num_timesteps=True,
    )

    # Save the FINAL model in addition to the eval best_model.
    model.save(os.path.join(models_dir, "final_model"))
    vec_normalize = model.get_vec_normalize_env()
    if vec_normalize is not None:
        vec_normalize.save(vecnormalize_path_for(os.path.join(models_dir, "final_model.zip")))
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
