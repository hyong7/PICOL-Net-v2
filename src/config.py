from dataclasses import dataclass, field


@dataclass
class SSPConfig:
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class PredictorConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1


@dataclass
class ECMConfig:
    eps: float = 1e-06
    current_sign: float = 1.0
    r_min: float = 1e-05
    c_min: float = 0.001
    r_base: float = 0.001
    c_base: float = 10000.0


@dataclass
class ModelConfig:
    feature_dim: int
    window: int = 50
    ssp: SSPConfig = field(default_factory=SSPConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    ecm: ECMConfig = field(default_factory=ECMConfig)
