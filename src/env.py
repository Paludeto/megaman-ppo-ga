"""
Module: src.env
Description: Provides utilities for creating and vectorizing the Mega Man NES environment
             with the custom wrappers for the Robot Master boss fights.
"""

import os

import stable_retro
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

try:
    from src.wrappers import ActionSkipWrapper, FrameskipWrapper, BossWrapper, StickyFrames, WarpFrame
except ImportError:
    from wrappers import ActionSkipWrapper, FrameskipWrapper, BossWrapper, StickyFrames, WarpFrame

# Custom integrations directory (at the project root). Registered ONCE per process: SubprocVecEnv
# re-imports this module in every worker, so each process registers its own.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CUSTOM_INTEGRATIONS_DIR = os.path.join(_PROJECT_ROOT, "custom_integrations")
_integrations_registered = False
_verified_roms = set()


def _register_custom_integrations():
    """Register the custom integrations path with stable-retro (idempotent per process)."""
    global _integrations_registered
    if not _integrations_registered:
        stable_retro.data.add_custom_integration(_CUSTOM_INTEGRATIONS_DIR)
        _integrations_registered = True


def _verify_rom(game):
    """
    Check the ROM of `game` against its rom.sha (once per process and game).

    stable_retro.make() only checks that rom.nes exists, so a ROM from another revision would
    load silently, with the RAM addresses in data.json pointing at the wrong bytes. The
    stable-retro convention is the SHA-1 of the file WITHOUT the 16-byte iNES header.
    """
    if game in _verified_roms:
        return
    inttype = stable_retro.data.Integrations.CUSTOM_ONLY
    try:
        errors = stable_retro.data.verify_hash(game, inttype)
    except FileNotFoundError:
        errors = [(game, "rom.nes not found")]
    if errors:
        try:
            with open(stable_retro.data.get_file_path(game, "rom.sha", inttype)) as f:
                expected = f.read().split()[0]
        except (OSError, TypeError, IndexError):
            expected = "<see rom.sha in the integration folder>"
        raise RuntimeError(
            f"ROM check failed for '{game}': rom.nes is missing or is a different revision/dump "
            f"({errors[0][1]}). Expected SHA-1 (without the 16-byte iNES header): {expected}. "
            f"Place your copy at custom_integrations/{game}/rom.nes and check it with: "
            f"tail -c +17 rom.nes | sha1sum"
        )
    _verified_roms.add(game)


def make_env(state, game="MegaMan-v1-Nes", d=0.05, win_bonus=0.0, damage_penalty_mult=1.0, frameskip=4, max_episode_frames=7200, sticky=0.0):
    """
    Creates a single instance of the Mega Man retro environment wrapped with the custom
    RL preprocessing steps.

    Parameters:
        state (str): Name of the initial save state to load on reset.
        game (str): Name of the game in stable-retro integrations.
        d (float): Flat reward scale passed to BossWrapper for hits/damage.
        sticky (float): per-frame sticky-action probability (Machado et al. 2018), applied BELOW
            ActionSkipWrapper like src/eval.py does. 0 (default) = the deterministic env, unchanged.
            The RNG is seeded by reset(seed=...) (SB3 seeds worker i with seed + i).

    Returns:
        gym.Env: Wrapped Mega Man retro environment instance.
    """
    _register_custom_integrations()
    _verify_rom(game)

    env = stable_retro.make(
        game=game,
        state=state,
        inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=stable_retro.Actions.MULTI_DISCRETE,
        obs_type=stable_retro.Observations.IMAGE,
        render_mode=None,
    )

    if sticky > 0.0:
        env = StickyFrames(env, sticky)
    env = ActionSkipWrapper(env)
    env = FrameskipWrapper(env, skip=frameskip)
    # max_episode_steps counts agent steps (skip=4 -> 1800 agent steps for 7200 frames), and a step
    # can span more than `skip` frames during the blink extension.
    env = TimeLimit(env, max_episode_steps=max_episode_frames // frameskip)
    env = BossWrapper(env, d=d, win_bonus=win_bonus, damage_penalty_mult=damage_penalty_mult)
    env = WarpFrame(env)
    return env


def make_venv(state, n_envs=8, game="MegaMan-v1-Nes", force_subproc=False, d=0.05, win_bonus=0.0, damage_penalty_mult=1.0, frameskip=4, norm_reward=False, max_episode_frames=7200, sticky=0.0):
    """
    Creates a vectorized environment with multiple parallel instances of make_env.
    Applies Frame Stacking (4 frames), Transposition to PyTorch format (C, H, W),
    and sets up basic metrics monitoring.

    Parameters:
        state (str): Name of the initial save state to load.
        n_envs (int): Number of parallel environments.
        game (str): Name of the game in stable-retro.
        force_subproc (bool): If True, forces using SubprocVecEnv even for n_envs=1.
        d (float): Flat reward scale passed through to BossWrapper for hits/damage.
        sticky (float): per-frame sticky-action probability of every env (see make_env).

    Returns:
        VecMonitor: Vectorized, stacked, transposed, and monitored environment wrapper.
    """
    env_kwargs = dict(
        game=game, state=state, d=d, win_bonus=win_bonus, damage_penalty_mult=damage_penalty_mult,
        frameskip=frameskip, max_episode_frames=max_episode_frames, sticky=sticky,
    )

    def make_thunk():
        return lambda: make_env(**env_kwargs)

    if n_envs > 1 or force_subproc:
        venv = SubprocVecEnv([make_thunk() for _ in range(n_envs)])
    else:
        venv = DummyVecEnv([make_thunk()])

    # Stack 4 frames, transpose image to (C, H, W) and monitor hp/boss_hp
    venv = VecFrameStack(venv, n_stack=4, channels_order='last')
    venv = VecTransposeImage(venv)
    venv = VecMonitor(venv, info_keywords=("hp", "boss_hp"))
    # Reward normalization (running std). Reward only (norm_obs=False), so it does not affect the
    # policy's observations/actions — it is eval-safe and the stats can be fresh per stage.
    if norm_reward:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)
    return venv
