"""
Shared pytest setup. Run from the project root: `pytest`.

Every test that needs the emulator requests the `rom_available` fixture (directly or through
`open_env` / `open_raw_env`), which skips the test when the user-supplied, git-ignored ROM is missing.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ROM_PATH = os.path.join(ROOT, "custom_integrations", "MegaMan-v1-Nes", "rom.nes")

STATES = ["Bombman-boss", "Cutman-boss", "Gutsman-boss", "Elecman-boss", "Iceman-boss", "Fireman-boss"]


@pytest.fixture(scope="session")
def rom_available():
    """Skip the requesting test if custom_integrations/MegaMan-v1-Nes/rom.nes does not exist."""
    if not os.path.isfile(ROM_PATH):
        pytest.skip(f"ROM not found at {ROM_PATH}")
    return ROM_PATH


@pytest.fixture
def open_env(rom_available):
    """Factory for wrapped envs (src.env.make_env), closed at teardown.

    The emulator allows one instance per process, so the factory closes the env it handed out
    previously before creating a new one.
    """
    from src.env import make_env
    opened = []

    def _open(**kwargs):
        while opened:
            opened.pop().close()
        env = make_env(**kwargs)
        opened.append(env)
        return env

    yield _open
    while opened:
        opened.pop().close()


@pytest.fixture
def open_raw_env(rom_available):
    """Factory for the bare stable-retro env (MULTI_DISCRETE, no wrappers), closed at teardown."""
    import stable_retro
    from src.env import _register_custom_integrations
    _register_custom_integrations()
    opened = []

    def _open(state):
        while opened:
            opened.pop().close()
        env = stable_retro.make(game="MegaMan-v1-Nes", state=state,
                                inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
                                use_restricted_actions=stable_retro.Actions.MULTI_DISCRETE,
                                obs_type=stable_retro.Observations.IMAGE, render_mode=None)
        opened.append(env)
        return env

    yield _open
    while opened:
        opened.pop().close()


@pytest.fixture
def require_rom(rom_available):
    """Alias used by the frameskip tests: skip when the ROM is absent."""
    return rom_available
