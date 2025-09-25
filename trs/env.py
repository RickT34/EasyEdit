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
    def get_layers_scattered(self, step:int):
        return itertools.chain(range(0, self.layers, step), [self.layers-1] if self.layers % step != 1 else [])
        


ModelLLaMA3 = ModelEnv("llama3-8b", "models/LLama-3-8B-Instruct", 32)
ModelQwen2p5 = ModelEnv("qwen2.5-7b", "models/Qwen2.5-7B-Instruct", 28)


@dataclasses.dataclass
class DatasetEnv:
    path: str
    range: str = 'All'

    @staticmethod
    def get_ds_name(data_json: str):
        ds_name = os.path.basename(data_json)
        ds_name = ds_name[: ds_name.rfind(".")]
        return ds_name

    def __post_init__(self):
        self.name = self.get_ds_name(self.path)

DatasetMQCF2hop = list(DatasetEnv('') for i in range(2))
DatasetMQCF3hop = list(DatasetEnv('') for i in range(3))


@dataclasses.dataclass
class AlgoEnv:
    name: str
    hparams_dir: str
    
def EasyEditAlgo(name):
    return AlgoEnv(name, f"../hparams/{name}")




@dataclasses.dataclass
class ExpEnv:
    model: ModelEnv
    dataset: DatasetEnv
    algo: AlgoEnv
    label: str
    outputs_dir: str
    tags: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.filename = f"{self.dataset}.json"

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

    def __str__(self):
        return "ExpEnv:\n" + "\n".join(
            [f"    {k}: {v}" for k, v in dataclasses.asdict(self).items()]
        )


ExpHistoryDir = 'exp_history'
def dumpEnvs(envs:list[ExpEnv]):
    l = list(dataclasses.asdict(e) for e in envs)
    os.makedirs(ExpHistoryDir, exist_ok=True)
    num = 1
    def _get_filename():
        return f'exp_{num}.json'
    while True:
        if not os.path.exists(os.path.join(ExpHistoryDir, _get_filename())):
            break
        num += 1
    with open(os.path.join(ExpHistoryDir, _get_filename()), 'w') as f:
        json.dump(l, f, indent=4)
        
def loadEnvs(path:str):
    with open(path, 'r') as f:
        l = json.load(f)
    return [ExpEnv(**e) for e in l]
             
