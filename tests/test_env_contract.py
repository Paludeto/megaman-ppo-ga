"""
Environment contract tests: action layout, initial RAM of the 6 save states and the
event-indicator reward. All of them need the ROM and are skipped without it.
"""

import numpy as np
import pytest

from conftest import STATES

D = 0.05
MULT = 2.0
NOOP = np.array([0, 0, 0])


def test_action_layout_is_b_dpad_a(open_raw_env):
    """MultiDiscrete([2,5,2]) = [B, D-pad(none,UP,DOWN,LEFT,RIGHT), A], not scenario.json order."""
    env = open_raw_env("Gutsman-boss")
    assert list(env.action_space.nvec) == [2, 5, 2]
    names = [[[env.buttons[k] for k in range(len(env.buttons)) if (c >> k) & 1] for c in combos]
             for combos in env.button_combos]
    assert names == [
        [[], ["B"]],
        [[], ["UP"], ["DOWN"], ["LEFT"], ["RIGHT"]],
        [[], ["A"]],
    ]


# The 6 save states were NOT recorded in a uniform way. health=28, boss_health=28 and
# weapon_energy=28 everywhere, but:
#   - blink_counter=3 (Mega Man still in post-damage blink) at the start of Bombman and Cutman, 0 elsewhere;
# This table pins that heterogeneity: recreating a state changes a row and must be a conscious edit.
INITIAL_RAM = {
    "Bombman-boss":     dict(lives=9, blink_counter=3, x=136, y=180),
    "Cutman-boss":      dict(lives=9, blink_counter=3, x=24, y=148),
    "Gutsman-boss":     dict(lives=9, blink_counter=0, x=24, y=148),
    "Elecman-boss":     dict(lives=9, blink_counter=0, x=136, y=180),
    "Iceman-boss":      dict(lives=9, blink_counter=0, x=27, y=180),
    "Fireman-boss":     dict(lives=9, blink_counter=0, x=24, y=180),
}


def test_initial_ram_table_covers_all_states():
    assert set(INITIAL_RAM) == set(STATES)


@pytest.mark.parametrize("state", STATES)
def test_initial_ram_of_save_states(open_raw_env, state):
    """RAM right after reset, per save state."""
    env = open_raw_env(state)
    env.reset()
    exp = INITIAL_RAM[state]

    def lookup(name):
        return int(env.data.lookup_value(name))

    assert lookup("health") == 28
    assert lookup("boss_health") == 28
    assert lookup("weapon_energy") == 28
    for name in ("lives", "blink_counter", "x", "y"):
        assert lookup(name) == exp[name], name


def make_reward_env(open_env, state="Gutsman-boss"):
    # Training-like knobs (m=2) but no win bonus / shaping / survival: only the two event terms.
    return open_env(state=state, d=D, damage_penalty_mult=MULT, win_bonus=0.0)


def test_reward_hit_and_damage_events(open_env):
    """+d for a step with a boss_hp drop, -m*d for a step with an hp drop, both in the same step."""
    env = make_reward_env(open_env)
    data = env.unwrapped.data
    env.reset()

    _, r, terminated, _, info = env.step(NOOP)
    assert r == 0.0 and not terminated

    data.set_value("boss_health", 27)                     # boss hit (any size = 1 event)
    _, r, terminated, _, info = env.step(NOOP)
    assert r == pytest.approx(D) and info['boss_hp'] == 27 and not terminated

    data.set_value("health", 24)                          # damage: 4 HP, still a single event
    _, r, terminated, _, info = env.step(NOOP)
    assert r == pytest.approx(-MULT * D) and info['hp'] == 24 and not terminated

    data.set_value("boss_health", 25)                     # a 2-HP drop is still +d, not +2d
    data.set_value("health", 20)
    _, r, terminated, _, info = env.step(NOOP)
    assert r == pytest.approx(D - MULT * D) and info['boss_hp'] == 25 and info['hp'] == 20


def test_reward_matches_event_indicators_on_random_rollout(open_env):
    """For every step of a random rollout: r == d*[boss_hp fell] - m*d*[hp fell]."""
    env = make_reward_env(open_env)
    rng = np.random.default_rng(0)
    hits = dmg = 0
    for _ in range(4):
        _, info = env.reset()
        prev_hp, prev_boss = info['hp'], info['boss_hp']
        for _ in range(1800):
            _, r, terminated, truncated, info = env.step(rng.integers([0, 0, 0], [2, 5, 2]))
            hit, hurt = info['boss_hp'] < prev_boss, info['hp'] < prev_hp
            assert r == pytest.approx(D * hit - MULT * D * hurt)
            hits += hit
            dmg += hurt
            prev_hp, prev_boss = info['hp'], info['boss_hp']
            if terminated or truncated:
                break
    assert hits > 0 and dmg > 0     # the rollout exercised both events
