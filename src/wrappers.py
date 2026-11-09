"""
Module: src.wrappers
Description: Custom Gym wrappers to process observations, adjust fire cadences,
             handle dynamic frameskipping during blink states, and compute flat rewards
             for Mega Man 1 boss battles.
"""

import gymnasium as gym
import numpy as np
import cv2

class StickyFrames(gym.Wrapper):
    """
    Sticky actions per emulator frame (Machado et al. 2018): at every frame, with probability
    `p` the action of the PREVIOUS frame is repeated instead of the requested one. The "previous
    action" is the one actually executed, so a sticky streak keeps repeating the same action.
    With p == 0 it is a pass-through that never draws from the RNG.

    It must sit directly on the raw stable-retro env, BELOW ActionSkipWrapper/FrameskipWrapper, so
    stickiness acts on every emulator frame. `rng` may be replaced per episode (src/eval.py does);
    reset(seed=s) reseeds it, which is how SB3 seeds each VecEnv worker (seed + rank) in training.
    """

    def __init__(self, env, p, rng=None):
        super().__init__(env)
        self.p = float(p)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.prev = np.zeros(3, dtype=int)

    def reset(self, **kwargs):
        if kwargs.get("seed") is not None:
            self.rng = np.random.default_rng(kwargs["seed"])
        self.prev = np.zeros(3, dtype=int)
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.array(action)
        if self.p > 0.0 and self.rng.random() < self.p:
            action = self.prev.copy()
        self.prev = action.copy()
        return self.env.step(action)


class ActionSkipWrapper(gym.Wrapper):
    """
    Alternates sending the shoot input (B button) when it is held by the agent,
    to respect the NES Mega Man firing cadence and avoid wasted inputs.
    """
    def __init__(self, env):
        """
        Initializes the action skip wrapper.

        Parameters:
            env (gym.Env): The underlying Gym environment.
        """
        super().__init__(env)
        self.b_state = 0

    def reset(self, **kwargs):
        """
        Resets the internal toggle state for the shoot button and the environment.

        Returns:
            tuple: Initial observation and environment info dictionary.
        """
        self.b_state = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Processes a single step in the environment. If the agent holds the shoot (B)
        button, alternates releasing it on alternate frames to ensure projectiles fire.

        Parameters:
            action (np.ndarray or list): MultiDiscrete actions of shape (3,).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # MultiDiscrete([2, 5, 2]) action for this integration:
        #   action[0] = B (SHOOT), action[1] = D-pad, action[2] = A (JUMP).
        # The NES firing cadence requires releasing+pressing B for each projectile -> we toggle B
        # (action[0]) so a policy that "holds B" can still fire in bursts. A (jump) is NOT toggled.
        modified_action = action.copy() if hasattr(action, 'copy') else list(action)
        if modified_action[0] == 1:
            if self.b_state == 1:
                modified_action[0] = 0
            self.b_state = 1 - self.b_state
        else:
            self.b_state = 0
        return self.env.step(modified_action)


class FrameskipWrapper(gym.Wrapper):
    """
    Repeats the action for `skip` frames. If the agent is flashing after taking damage, it keeps
    repeating the action (up to 60 extra frames) while `blink_counter % 4 in {1, 2}`. This is only a
    rough proxy for on-screen visibility, the step length varies, and the agent has no control while
    the extension runs.
    """

    def __init__(self, env, skip=4):
        """
        Initializes the frameskip wrapper.

        Parameters:
            env (gym.Env): The underlying Gym environment.
            skip (int): The number of base frames to skip.
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """
        Applies frameskip by repeating the action for `skip` steps, and then continues
        skipping frames dynamically while the player character is invisible (blinking).

        Parameters:
            action (np.ndarray or list): Action vector to repeat.

        Returns:
            tuple: (observation, aggregated_reward, terminated, truncated, final_info)
        """
        total_reward = 0.0
        terminated = False
        truncated = False

        # Step for the fixed frame-skip amount
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info

        # Continue stepping if Mega Man is in an invisible blink state.
        # Safety cap: never lock the loop if the blink never resolves to visible.
        max_blink_steps = 60
        blink_steps = 0
        while True:
            blink_counter = info.get('blink_counter', 0)
            # Periodic NES Mega Man blink pattern (invisible if remainder is 1 or 2)
            visibility = 'i' if (blink_counter % 4 in (1, 2)) and blink_counter > 0 else 'v'
            if visibility == 'v':
                break

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

            blink_steps += 1
            if blink_steps >= max_blink_steps:
                break

        return obs, total_reward, terminated, truncated, info


class BossWrapper(gym.Wrapper):
    """
    Custom wrapper for the boss fight scenario. Implements the flat reward function
    and the termination condition.
    """
    def __init__(self, env, d=0.05, win_bonus=0.0, damage_penalty_mult=1.0):
        """
        Initializes the boss reward and termination wrapper.

        Parameters:
            env (gym.Env): The underlying Gym environment.
            d (float): Flat reward scale for dealing damage and penalty for taking damage.
            win_bonus (float): Terminal bonus added to the reward when the boss dies (boss_health=0).
            damage_penalty_mult (float): Multiplier on the damage-taken penalty.
        """
        super().__init__(env)
        self.d = d
        self.win_bonus = win_bonus
        self.damage_penalty_mult = damage_penalty_mult
        self.prev_health = 0
        self.prev_boss_health = 0
        self.prev_lives = 0
        self.prev_weapon_energy = 28

    @staticmethod
    def _info_dict(health, boss_health, lives, weapon_energy, info):
        """Build the info dict exposed to the agent (same format in reset and step)."""
        return {
            'hp': health,
            'boss_hp': boss_health,
            'lives': lives,
            'x': info.get('x', 0),
            'y': info.get('y', 0),
            'screen': info.get('screen', 0),
            'weapon_energy': weapon_energy,
        }

    def reset(self, **kwargs):
        """
        Resets environment tracking variables for health, lives, and boss health.

        Returns:
            tuple: (observation, custom_info)
        """
        obs, info = self.env.reset(**kwargs)
        # IMPORTANT: retro's info on reset is usually EMPTY (no 'lives', 'boss_health', etc.). Trusting
        # info.get(name, default) with a wrong default for `lives` would end the 1st step with a spurious
        # "lost a life" termination. We read the REAL values from RAM.
        def _ram(name, default):
            try:
                return int(self.unwrapped.data.lookup_value(name))
            except KeyError:  # variable not defined in the integration's data.json
                return int(info.get(name, default))
        self.prev_health = _ram('health', 28)
        self.prev_boss_health = _ram('boss_health', 28)
        self.prev_lives = _ram('lives', 9)
        self.prev_weapon_energy = _ram('weapon_energy', 28)

        custom_info = self._info_dict(self.prev_health, self.prev_boss_health, self.prev_lives,
                                      self.prev_weapon_energy, info)
        return obs, custom_info

    def step(self, action):
        """
        Computes custom flat reward based on damage events, and returns early episode
        termination when boss or player health drops to zero, lives decrease, or spikes are hit.

        Parameters:
            action (np.ndarray): Action to take.

        Returns:
            tuple: (observation, reward, terminated, truncated, custom_info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Use the previous value as default if a RAM read fails/is missing, avoiding spurious
        # terminations and win bonuses (default 0 would trigger health=0 / boss=0).
        cur_health = info.get('health', self.prev_health)
        cur_boss_health = info.get('boss_health', self.prev_boss_health)
        cur_lives = info.get('lives', self.prev_lives)
        touching_obj_top = info.get('touching_obj_top', 0)
        touching_obj_side = info.get('touching_obj_side', 0)
        cur_weapon_energy = info.get('weapon_energy', self.prev_weapon_energy)

        # Reward function:
        #   +d                  per hit event on the boss
        #   -d*damage_mult      per damage-taken event
        #   +win_bonus          terminal, on killing the boss
        reward_boss = 0.0

        boss_took_damage = self.prev_boss_health > cur_boss_health
        if boss_took_damage:
            reward_boss += self.d
        if self.prev_health > cur_health:
            reward_boss -= self.d * self.damage_penalty_mult

        # Determine termination
        custom_terminated = False
        if cur_health == 0:
            custom_terminated = True
        elif cur_lives < self.prev_lives:
            custom_terminated = True
        elif touching_obj_top == 3 or touching_obj_side == 3:
            custom_terminated = True
        elif cur_boss_health == 0:
            custom_terminated = True
            reward_boss += self.win_bonus  # terminal win bonus (sparse)
        elif cur_weapon_energy == 0 and cur_boss_health > 0:
            # Early termination if ammo runs out and the boss is still alive
            custom_terminated = True

        # Save previous values
        self.prev_health = cur_health
        self.prev_boss_health = cur_boss_health
        self.prev_lives = cur_lives
        self.prev_weapon_energy = cur_weapon_energy

        custom_info = self._info_dict(cur_health, cur_boss_health, cur_lives,
                                      cur_weapon_energy, info)

        return obs, reward_boss, custom_terminated, truncated, custom_info


class WarpFrame(gym.ObservationWrapper):
    """
    Converts RGB observations to grayscale and resizes them to 84x84.
    """
    def __init__(self, env):
        """
        Initializes the frame preprocessing wrapper.

        Parameters:
            env (gym.Env): The underlying Gym environment.
        """
        super().__init__(env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        """
        Preprocesses raw RGB frames to 84x84 grayscale.

        Parameters:
            obs (np.ndarray): Raw RGB observation.

        Returns:
            np.ndarray: Grayscale resized observation of shape (84, 84, 1).
        """
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)
