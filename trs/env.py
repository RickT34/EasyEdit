from pathlib import Path
import dataclasses
import os
import json
import itertools


@dataclasses.dataclass
class ModelEnv:
    name: str
    path: str
    layers: int
    tags: dict = dataclasses.field(default_factory=dict)

    def get_layers_scattered(self, step: int):
        return itertools.chain(
            range(0, self.layers, step),
            [self.layers - 1] if self.layers % step != 1 else [],
        )


ModelLLaMA3 = ModelEnv("llama3-8b", "models/LLama-3-8B-Instruct", 32)
ModelQwen2p5 = ModelEnv("qwen2.5-7b", "models/Qwen2.5-7B-Instruct", 28)


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


DatasetsMQCF2hop800 = list(
    DatasetEnv(f"dataset/mq_cf_sample800_2hop/mq_cf_sample800_2hop{i+1}.json", tags={"loc": i})
    for i in range(2)
)
DatasetsMQCF2hop200 = list(
    DatasetEnv(f"dataset/mq_cf_sample200_2hop/mq_cf_sample200_2hop{i+1}.json", tags={"loc": i})
    for i in range(2)
)
DatasetsMQCF3hop100 = list(
    DatasetEnv(f"dataset/mq_cf_sample100_3hop/mq_cf_sample100_3hop{i+1}.json", tags={"loc": i})
    for i in range(3)
)


@dataclasses.dataclass
class AlgoEnv:
    name: str
    hparams_dir: str
    hparams_rewrite: dict = dataclasses.field(default_factory=dict)
    tags: dict = dataclasses.field(default_factory=dict)


def EasyEditAlgo(name):
    hparams_rewrite = {
        "FT-M": {"objective_optimization": "target_new"},
        "FT-L": {"objective_optimization": "prompt_last"},
    }.get(name, {})

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


ExpHistoryDir = Path("exp_history")


def dumpEnv(env: ExpEnv, path):
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(env), f, indent=4)


def dumpEnvs(envs: list[ExpEnv]):
    os.makedirs(ExpHistoryDir, exist_ok=True)
    num = 1

    def _get_name():
        return ExpHistoryDir / f"exp_{num}"

    while _get_name().exists():
        num += 1
    dir = _get_name()
    os.makedirs(dir)
    paths = []
    for i, e in enumerate(envs):
        path = dir / f"env_{i+1}.json"
        paths.append(path)
        dumpEnv(e, path)
    return paths


def loadEnv(path):
    with open(path, "r") as f:
        d = json.load(f)
    return ExpEnv(**d)


def envCopy(env, cls):
    return cls(**dataclasses.asdict(env))
