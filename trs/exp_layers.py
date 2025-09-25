import itertools
import lazyexp
from env import *
from utils import make_cmd_maker

DEVICE_FREE = [7]


def mk_exp_cmd_addon(env: ExpEnv, device):
    return [
        "--rewrite_hparams",
        f"dict(layers=[{env.tags['layer']}])",
    ]


cmd_maker = make_cmd_maker(mk_exp_cmd_addon)


def env_maker(layer: int, algo: str, data_json: str, model: ModelEnv):
    label = f"layer{layer}"
    return ExpEnv(
        model.model_name,
        get_ds_name(data_json),
        algo,
        label,
        f"All",
        f"outputs_layer",
        {"data_json": data_json, "model_path": model.model_path, "layer": layer},
    )


if __name__ == "__main__":
    params1 = list(
        itertools.product(
            list(range(0, 32, 4))+[31],
            ["FT-M"],
            [
                "dataset/mq_cf_sample200_2hop/mq_cf_sample200_2hop1.json",
                "dataset/mq_cf_sample200_2hop/mq_cf_sample200_2hop2.json",
            ],
            [ModelLLaMA3],
        )
    )
    params3 = list(
        itertools.product(
            list(range(0, 28, 4))+[27],
            ["FT-M"],
            [
                "dataset/mq_cf_sample100_3hop/mq_cf_sample100_3hop1.json",
                "dataset/mq_cf_sample100_3hop/mq_cf_sample100_3hop2.json",
                "dataset/mq_cf_sample100_3hop/mq_cf_sample100_3hop3.json",
            ],
            ["qwen2.5-7b"],
        )
    )
    envs = list(itertools.starmap(env_maker, params1))
    lazyexp.run_exps(envs, DEVICE_FREE, cmd_maker)
