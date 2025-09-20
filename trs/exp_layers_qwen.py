import lazyexp

MODEL_PATH = "models/Qwen2.5-7B-Instruct"
MODEL_NAME = "qwen2.5-7b"
DEVICE_FREE = [7]
ALGOs = ["FT-M"]
LOG_DIR = "explogs/nhop_layers"


def mk_exp_cmd(algo: str, data_json: str, label: str, device: int, ds_range: str, layer:int):
    return [
        "./lazyeditor.py",
        "--editing_method",
        algo,
        "--model_name",
        MODEL_NAME,
        "--device",
        str(device),
        "--ds_range",
        ds_range,
        "--label",
        label,
        "--model_path",
        MODEL_PATH,
        "--outputs_dir",
        "otuputs_layer",
        "--data_json",
        data_json,
        "--rewrite_hparams",
        f"dict(layers=[{layer}])"
    ]


if __name__ == "__main__":
    for algo in ALGOs:
        for n in [1, 2]:
            data_json = f"dataset/mq_cf_sample200_2hop{n}.json"
            layers = list(range(0, 32, 4)) + [31]
            for l in layers:
                label = f"layer{l}"
                def cmd_maker(i):
                    return mk_exp_cmd(algo, data_json, label, DEVICE_FREE[i], f"{i+1}q{len(DEVICE_FREE)}" if len(DEVICE_FREE)>1 else "All", l)
                lazyexp.run_exp(LOG_DIR, algo, label, DEVICE_FREE, cmd_maker)

