from pathlib import Path
import dataclasses
import os
import json
import itertools
import math


@dataclasses.dataclass
class ModelEnv:
    name: str
    path: str
    layers: int
    tags: dict = dataclasses.field(default_factory=dict)

    def get_layers_scattered(self, step: int, count:int = 1):
        i = 0
        while i < self.layers:
            yield list(range(i, min(self.layers, i + count)))
            i += step
        if i - step + count < self.layers:
            yield [self.layers-1]


ModelLLaMA3 = ModelEnv("llama3-8b", "data/models/LLama-3-8B-Instruct", 32)
ModelQwen2p5 = ModelEnv("qwen2.5-7b", "data/models/Qwen2.5-7B-Instruct", 28)

@dataclasses.dataclass
class DatasetEnv:
    path: str
    range: str = "All"
    tags: dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def get_ds_name(data_json: str):
        ds_name = os.path.basename(data_json)
        ds_name = ds_name[: ds_name.rfind(".")]
        return ds_name

    def __post_init__(self):
        self.name = self.get_ds_name(self.path)
        if self.range == "1q1":
            self.range = "All"
            
    @staticmethod
    def ds_split(l:int, m:int, n:int):
        k = math.trunc(l/n)
        return k*(m-1), l if m==n else k*m
    
    def read(self):
        return json.load(open(self.path, "r"))


DatasetsMQCF2hop800 = list(
    DatasetEnv(f"data/dataset/mq_cf_sample800_2hop/mq_cf_sample800_2hop{i+1}.json", tags={"loc": i})
    for i in range(2)
)
DatasetsMQCF2hop800inv = list(
    DatasetEnv(f"data/dataset/mq_cf_sample800_2hopinv/mq_cf_sample800_2hopinv{i+1}.json", tags={"loc": i})
    for i in range(2)
)
DatasetsMQCF2hop200 = list(
    DatasetEnv(f"data/dataset/mq_cf_sample200_2hop/mq_cf_sample200_2hop{i+1}.json", tags={"loc": i})
    for i in range(2)
)
DatasetsMQCF3hop100 = list(
    DatasetEnv(f"data/dataset/mq_cf_sample100_3hop/mq_cf_sample100_3hop{i+1}.json", tags={"loc": i})
    for i in range(3)
)
DatasetMQCF2chop200 = DatasetEnv("data/dataset/mq_cf_sample200_2chop_2.json")
DatasetMQCFAllEdges = DatasetEnv("data/dataset/mq_cf_all_edges.json")

@dataclasses.dataclass
class AlgoEnv:
    name: str
    hparams_dir: str
    hparams_rewrite: dict = dataclasses.field(default_factory=dict)
    tags: dict = dataclasses.field(default_factory=dict)


def EasyEditAlgo(name:str):
    hparams_rewrite = {
        "FT-M": {"objective_optimization": "target_new"},
        "FT-L": {"objective_optimization": "prompt_last"},
        "ROME": {"mom2_adjustment": True},
    }.get(name, {})
    name = {
        "FT-M": "FT",
        "FT-L": "FT",
    }.get(name, name)

    return AlgoEnv(name, f"../hparams/{name}", hparams_rewrite)


@dataclasses.dataclass
class ExpEnv:
    model: ModelEnv
    dataset: DatasetEnv
    algo: AlgoEnv
    label: str
    outputs_dir: str
    tags: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        for k, c in [
            ("model", ModelEnv),
            ("dataset", DatasetEnv),
            ("algo", AlgoEnv),
        ]:
            v = getattr(self, k)
            if not isinstance(v, c):
                setattr(self, k, c(**v))

        self.filename = f"{self.dataset.range}.json"

    def get_prefile_path(self, prefiles_dir: str):
        prefiledir = Path(prefiles_dir) / self.model.name / self.dataset.name
        prefiledir.mkdir(parents=True, exist_ok=True)
        pre_file = prefiledir / self.filename
        return pre_file

    def get_output_path(self):
        outputdir = (
            Path(self.outputs_dir)
            / self.model.name
            / self.dataset.name
            / self.label
            / self.algo.name
        )
        outputdir.mkdir(parents=True, exist_ok=True)
        output_file = outputdir / self.filename
        return output_file
    
    def to_json(self):
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(json_str):
        d = json.loads(json_str)
        return ExpEnv(**d)
    
    def dump(self, path):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            d = json.load(f)
        return ExpEnv(**d)


ExpHistoryDir = Path("exp_history")

def dumpEnvs(envs: list[ExpEnv], name:str):
    path = ExpHistoryDir / f"{name}.json"
    #assert not path.exists(), f"exp {name} already exists"
    l = []
    for e in envs:
        l.append(dataclasses.asdict(e))
    return json.dump(l, open(path, "w"), indent=4)

def loadEnvs(name:str):
    path = ExpHistoryDir / f"{name}.json"
    l = json.load(open(path, "r"))
    envs = []
    for d in l:
        envs.append(ExpEnv(**d))
    return envs
    

def envCopy(env, cls):
    return cls(**dataclasses.asdict(env))
