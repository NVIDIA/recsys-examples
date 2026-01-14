from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class DataType(Enum):
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    INT1 = "int1"
    UINT8 = "uint8"
    UINT4 = "uint4"
    UINT2 = "uint2"


@dataclass
class MachineConfiguration:
    peak_flops: Dict[str, float] = field(default_factory=dict)
    bandwidth: float = 0.0
    ridge_point: float = field(init=False)

    def __post_init__(self):
        self.ridge_point = self.bandwidth / self.peak_flops
        for key in self.peak_flops.keys():
            assert DataType(key) in DataType.__members__.keys(), f"Invalid dtype: {key}"

    def __repr__(self):
        return f"MachineConfiguration(peak_flops={self.peak_flops}, bandwidth={self.bandwidth}, ridge_point={self.ridge_point})"
