from dataclasses import dataclass

@dataclass
class Paths:
    in_folder: str
    out_folder: str

@dataclass
class Params:
    sample_size: int
    model_ckpt:  str
    num_labels: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    run_name: str

@dataclass
class SteamConfigClass:
    paths: Paths
    params: Params
