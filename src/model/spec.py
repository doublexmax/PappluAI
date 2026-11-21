from dataclasses import dataclass
from typing import Mapping

CHECKPOINT_VERSION = 5
ARCHITECTURE = "card_mlp"


class ModelSpecError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    width: int = 128
    blocks: int = 3

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 8:
            raise ModelSpecError("model width must be an integer of at least 8")
        if type(self.blocks) is not int or self.blocks < 1:
            raise ModelSpecError("model blocks must be a positive integer")

    def to_dict(self) -> dict:
        return {"width": self.width, "blocks": self.blocks}

    @staticmethod
    def from_dict(data: Mapping[str, object]) -> "ModelSpec":
        if not isinstance(data, Mapping) or set(data) != {"width", "blocks"}:
            raise ModelSpecError("checkpoint model_spec must contain width and blocks for card_mlp v5")
        return ModelSpec(**data)
