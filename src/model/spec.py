from dataclasses import asdict, dataclass

CHECKPOINT_VERSION = 3
ARCHITECTURE = "dynamic_table"


@dataclass(frozen=True)
class ModelSpec:
    width: int = 64
    heads: int = 4

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 4:
            raise ValueError("model width must be an integer of at least 4")
        if type(self.heads) is not int or self.heads < 1 or self.width % self.heads:
            raise ValueError("model heads must divide width")

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "ModelSpec":
        if not isinstance(data, dict) or set(data) != {"width", "heads"}:
            raise ValueError("checkpoint model_spec must contain width and heads")
        return ModelSpec(**data)
