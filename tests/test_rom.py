import os
import shutil

import pytest

stable_retro = pytest.importorskip("stable_retro")

from src import env as env_module

_INTEGRATION = os.path.join(env_module._CUSTOM_INTEGRATIONS_DIR, "MegaMan-v1-Nes")
_ROM = os.path.join(_INTEGRATION, "rom.nes")

pytestmark = pytest.mark.skipif(not os.path.exists(_ROM), reason="rom.nes not available")


def test_correct_rom_passes():
    env_module._register_custom_integrations()
    errors = stable_retro.data.verify_hash("MegaMan-v1-Nes", stable_retro.data.Integrations.CUSTOM_ONLY)
    assert errors == []
    env_module._verify_rom("MegaMan-v1-Nes")  # must not raise


def test_altered_rom_raises(tmp_path):
    # Copy of the integration under another game name, with 1 byte changed in the PRG data.
    game = "MegaManBadRom-v1-Nes"
    bad_dir = tmp_path / game
    bad_dir.mkdir()
    shutil.copy(os.path.join(_INTEGRATION, "rom.sha"), bad_dir / "rom.sha")
    data = bytearray(open(_ROM, "rb").read())
    data[1000] ^= 0xFF
    (bad_dir / "rom.nes").write_bytes(bytes(data))
    stable_retro.data.add_custom_integration(str(tmp_path))

    with pytest.raises(RuntimeError, match="sha1sum") as exc:
        env_module.make_env(state="Gutsman-boss", game=game)
    assert "6047e52929dfe8ed4708d325766ccb8d3d583c7d" in str(exc.value)
    assert game not in env_module._verified_roms
