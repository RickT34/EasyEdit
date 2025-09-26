import itertools
import lazyexp
from env import *
from utils import make_cmd_maker

DEVICE_FREE = [6, 7]

cmd_maker = make_cmd_maker()


def env_maker(layer: int, algo: AlgoEnv, dataset: DatasetEnv, model: ModelEnv):
    label = f"layer{layer}"
    algo_new = dataclasses.replace(algo, hparams_rewrite={"layers": [layer]})
    return ExpEnv(model, dataset, algo_new, label, f"outputs_layer_rome")


if __name__ == "__main__":
    params1 = list(
        itertools.product(
            ModelLLaMA3.get_layers_scattered(4),
            [EasyEditAlgo("ROME")],
            [DatasetsMQCF2hop200[0]],
            [ModelLLaMA3],
        )
    )
    envs = list(itertools.starmap(env_maker, params1))
    lazyexp.run_exps(envs, DEVICE_FREE, cmd_maker)
