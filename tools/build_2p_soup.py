from __future__ import annotations

from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoints.io import atomic_copy
from src.model.network import average_checkpoints


SOURCES = (
    (str(ROOT / "checkpoints/model-3796caf24fc056a429d06244399a6ff525265f0d11d278b8f236a6ec0c90952f.pt"), 0.70),
    (str(ROOT / "checkpoints/model-414ee9808848b9c46447891eb1b6764e96e586ebecc49055d15bd7311dd36458.pt"), 0.30),
)


def main() -> None:
    recipe = "same_basin_70_30_heads"
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary) / (recipe + ".pt")
        _, _, sha256 = average_checkpoints(
            SOURCES, str(stage), meta={"recipe": recipe},
        )
        if not sha256.startswith("067db9016584"):
            raise RuntimeError("unexpected soup checkpoint SHA-256 " + sha256)
        snapshot = ROOT / "checkpoints" / ("model-" + sha256 + ".pt")
        champion = ROOT / "checkpoints/champion-2p-dynamic-table.pt"
        atomic_copy(stage, snapshot)
        atomic_copy(stage, champion)
    print("wrote %s and %s" % (champion, snapshot))
    print("sha256 " + sha256)


if __name__ == "__main__":
    main()
