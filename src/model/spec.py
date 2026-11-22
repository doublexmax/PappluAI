from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Self


CHECKPOINT_VERSION = 6
ARCHITECTURE = "game_transformer"
BYTE_VALUES = 256


class ModelSpecError(ValueError):
    pass


@dataclass(frozen=True)
class TokenSchema:
    name: str
    version: int
    fields: int
    suffix_tokens: int
    num_actions: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ModelSpecError("schema needs a name")
        if any(type(n) is not int or n < 1 for n in (self.version, self.fields, self.suffix_tokens)):
            raise ModelSpecError("schema version and dimensions must be positive integers")
        if type(self.num_actions) is not int or not 1 <= self.num_actions <= 63:
            raise ModelSpecError("schema must have 1..63 actions")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        if not isinstance(data, Mapping) or data.keys() != cls.__dataclass_fields__.keys():
            raise ModelSpecError("invalid token schema fields")
        return cls(**data)


@dataclass(frozen=True)
class ModelSpec:
    width: int = 192
    layers: int = 4
    heads: int = 6
    ffn: int = 512

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for n in (self.width, self.layers, self.heads, self.ffn)):
            raise ModelSpecError("model dimensions must be positive integers")
        if self.width % (2 * self.heads):
            raise ModelSpecError("width must divide into an even number of channels per head")

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        if not isinstance(data, Mapping) or data.keys() != cls.__dataclass_fields__.keys():
            raise ModelSpecError("invalid model spec fields")
        return cls(**data)
