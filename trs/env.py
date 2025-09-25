from pathlib import Path
from dataclasses import dataclass, asdict, field
import os

@dataclass
class ModelEnv:
    model_name: str
    model_path: str
    total_layers: int
    
ModelLLaMA3 = ModelEnv('llama3-8b', "models/LLama-3-8B-Instruct", 32)
ModelQwen2p5 = ModelEnv('qwen2.5-7b', "models/Qwen2.5-7B-Instruct", 28)


@dataclass
class DatasetEnv:
    ds_path: str
    ds_range: str
    @staticmethod
    def get_ds_name(data_json:str):
        ds_name = os.path.basename(data_json)
        ds_name = ds_name[: ds_name.rfind(".")]
        return ds_name

    def __post_init__(self):
        self.ds_name = self.get_ds_name(self.ds_path)

    

@dataclass
class ExpEnv:
    model_name: str
    ds_name: str
    algo: str
    label: str
    ds_range: str
    outputs_dir:str
    tags:dict=field(default_factory=dict)

    def __post_init__(self):
        self.filename = f"{self.ds_range}.json"

    def get_prefile_path(self, prefiles_dir: str):
        prefiledir = Path(prefiles_dir) / self.model_name / self.ds_name
        prefiledir.mkdir(parents=True, exist_ok=True)
        pre_file = prefiledir / self.filename
        return pre_file

    def get_output_path(self):
        outputdir = (
            Path(self.outputs_dir) / self.model_name / self.ds_name / self.label / self.algo
        )
        outputdir.mkdir(parents=True, exist_ok=True)
        output_file = outputdir / self.filename
        return output_file

    def __str__(self):
        return "ExpEnv:\n" + "\n".join(
            [f"    {k}: {v}" for k, v in asdict(self).items()]
        )

