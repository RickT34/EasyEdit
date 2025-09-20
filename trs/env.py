from pathlib import Path
from dataclasses import dataclass

@dataclass
class ExpEnv:
    model_name: str
    ds_name: str
    algo: str
    label: str
    ds_range: str

    def __post_init__(self):
        self.filename = f"{self.ds_range}.json"

    def get_prefile_path(self, prefiles_dir: str):
        prefiledir = Path(prefiles_dir) / self.model_name / self.ds_name
        prefiledir.mkdir(parents=True, exist_ok=True)
        pre_file = prefiledir / self.filename
        return pre_file

    def get_output_path(self, outputs_dir: str):
        outputdir = (
            Path(outputs_dir) / self.model_name / self.ds_name / self.label / self.algo
        )
        outputdir.mkdir(parents=True, exist_ok=True)
        output_file = outputdir / self.filename
        return output_file
