from dataclasses import asdict, dataclass

CHECKPOINT_VERSION = 4
ARCHITECTURE = "axial_table"


@dataclass(frozen=True)
class ModelSpec:
    width: int = 64
    heads: int = 4
    blocks: int = 2

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 8:
            raise ValueError("model width must be an integer of at least 8")
        if type(self.heads) is not int or self.heads < 1 or self.width % self.heads:
            raise ValueError("model heads must divide width")
        if type(self.blocks) is not int or self.blocks < 1:
            raise ValueError("model blocks must be a positive integer")

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "ModelSpec":
        if not isinstance(data, dict) or set(data) != {"width", "heads", "blocks"}:
            raise ValueError("checkpoint model_spec must contain width, heads, and blocks for axial_table v4")
        return ModelSpec(**data)
