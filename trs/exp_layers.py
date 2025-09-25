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


def env_maker(layer: int, algo: AlgoEnv, dataset: DatasetEnv, model: ModelEnv):
    label = f"layer{layer}"
    return ExpEnv(
        model,
        dataset,
        algo,
        label,
        f"outputs_layer",
        {"layer": layer},
    )


if __name__ == "__main__":
    params1 = list(
        itertools.product(
            ModelLLaMA3.get_layers_scattered(4),
            [EasyEditAlgo('ROME')],
            DatasetMQCF2hop,
            [ModelLLaMA3],
        )
    )
    envs = list(itertools.starmap(env_maker, params1))
    lazyexp.run_exps(envs, DEVICE_FREE, cmd_maker)
