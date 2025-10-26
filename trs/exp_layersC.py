import itertools
import lazyexp
from env import *
from utils import make_cmd_maker

DEVICE_FREE = [3, 5, 6, 7]

cmd_maker = make_cmd_maker()


def env_maker(layers: list, algo: AlgoEnv, dataset: DatasetEnv, model: ModelEnv):
    label = f"layer{layers[0]}"
    algo_new = dataclasses.replace(algo, hparams_rewrite={"layers": layers, **algo.hparams_rewrite})
    return ExpEnv(model, dataset, algo_new, label, f"outputs/layerC")


if __name__ == "__main__":
    params1 = list(
        itertools.product(
            ModelLLaMA3.get_layers_scattered(4),
            [EasyEditAlgo("ROME"), EasyEditAlgo("FT-M")],
            [DatasetMQCF2chop200],
            [ModelLLaMA3],
        )
    )
    params2 = list(
        itertools.product(
            ModelQwen2p5.get_layers_scattered(4),
            [EasyEditAlgo("ROME"), EasyEditAlgo("FT-M")],
            [DatasetMQCF2chop200],
            [ModelQwen2p5],
        )
    )
    params3 = list(
        itertools.product(
            ModelLLaMA3.get_layers_scattered(4, 5),
            [EasyEditAlgo("MEMIT")],
            [DatasetMQCF2chop200],
            [ModelLLaMA3],
        )
    )
    params4 = list(
        itertools.product(
            ModelQwen2p5.get_layers_scattered(4, 5),
            [EasyEditAlgo("MEMIT")],
            [DatasetMQCF2chop200],
            [ModelQwen2p5],
        )
    )
    envs = list(itertools.starmap(env_maker,params1+params2+params3+params4))
    lazyexp.run_exps("layersC", envs, DEVICE_FREE, cmd_maker)
