"""src/ has no silent broad `except`."""

import ast
import glob
import os

from conftest import ROOT


def test_no_silent_broad_except_in_src():
    """No bare `except:` and no `except Exception` whose body is just pass/continue."""
    for path in glob.glob(os.path.join(ROOT, "src", "*.py")):
        with open(path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            assert node.type is not None, f"bare except in {path}:{node.lineno}"
            broad = isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException")
            silent = all(isinstance(n, (ast.Pass, ast.Continue)) for n in node.body)
            assert not (broad and silent), f"silent broad except in {path}:{node.lineno}"
