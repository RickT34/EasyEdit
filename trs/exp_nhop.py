import lazyexp
from env import ExpEnv, get_ds_name
from utils import make_cmd_maker

OUTPUTS_DIR = "outputs"

MODEL_PATH = "models/Qwen2.5-7B-Instruct"
MODEL_NAME = "qwen2.5-7b"
DEVICE_FREE = [1,2,3,6]
ALGOs = ["AlphaEdit", "FT-M", "LoRA", "MEMIT", "ROME", "UltraEdit", "QLoRA"]
# ALGOs = ["FT-M"]

cmd_maker = make_cmd_maker()


if __name__ == "__main__":
    envs = []
    for algo in ALGOs:
        for n in [1, 2]:
            data_json = (
                f"/Data2/tangrui/EasyEdit/trs/dataset/mq_cf_sample800_2hop{n}.json"
            )
            label = f"mq_cf_sample800_2hop{n}"

            env = ExpEnv(
                MODEL_NAME,
                get_ds_name(data_json),
                algo,
                label,
                f"All",
                OUTPUTS_DIR,
                {"model_path": MODEL_PATH, "data_json": data_json}
            )
            envs.append(env)
    lazyexp.run_exps(envs, DEVICE_FREE, cmd_maker)
