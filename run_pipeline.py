"""Entry point of the Pressure Rating System: ``python run_pipeline.py [--from-stage STAGE]``.

Run it from the workspace root (Requirement 28, criterion 1). With no arguments it runs every
registered Pipeline_Stage in run order; ``--from-stage STAGE`` reruns STAGE and every later
stage from the saved outputs of the earlier ones (criterion 5). The exit code is 0 only when
the last stage completed and no Protected_File changed (criterion 8); the Run lifecycle and the
other exit codes are described in ``src/superr/pipeline.py``.

``src/`` is put first on ``sys.path``, so the Pipeline runs from this workspace's code with or
without the editable install. The workspace root is the folder that holds this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_SRC = str(ROOT / "src")
if sys.path[:1] != [_SRC]:
    sys.path.insert(0, _SRC)


def _main() -> int:
    from superr.pipeline import main

    return main(sys.argv[1:], root=ROOT)


if __name__ == "__main__":
    sys.exit(_main())
