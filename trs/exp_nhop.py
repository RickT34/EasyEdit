import lazyexp

LOG_DIR = "explogs/nhop_algo_qwen"

MODEL_PATH = "models/Qwen2.5-7B-Instruct"
MODEL_NAME = "qwen2.5-7b"
DEVICE_FREE = [6]
ALGOs = ["AlphaEdit", "FT-M", "LoRA", "MEMIT", "ROME", "UltraEdit", "QLoRA"]
# ALGOs = ["FT-M"]


def mk_exp_cmd(algo: str, data_json: str, label: str, device: int, ds_range: str):
    return [
        "./lazyeditor.py",
        "--editing_method", algo,
        "--model_name", MODEL_NAME,
        "--device", str(device),
        "--ds_range", ds_range,
        "--label", label,
        "--model_path", MODEL_PATH,
        "--data_json", data_json,
    ]


if __name__ == "__main__":
    for algo in ALGOs:
        for n in [1, 2]:
            data_json = f"/Data2/tangrui/EasyEdit/trs/dataset/mq_cf_sample800_2hop{n}.json"
            label = f"mq_cf_sample800_2hop{n}"
            def cmd_maker(i):
                return mk_exp_cmd(algo, data_json, label, DEVICE_FREE[i], f"{i+1}q{len(DEVICE_FREE)}")
            lazyexp.run_exp(LOG_DIR, algo, label, DEVICE_FREE, cmd_maker)

