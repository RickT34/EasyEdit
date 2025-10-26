import itertools
import lazyexp
from env import *
from utils import make_cmd_maker

DEVICE_FREE = [1, 3, 5, 6, 7]

cmd_maker = make_cmd_maker()


def env_maker(layer: int, algo: AlgoEnv, dataset: DatasetEnv, model: ModelEnv):
    label = f"layer{layer}"
    algo_new = dataclasses.replace(algo, hparams_rewrite={"layers": [layer], **algo.hparams_rewrite})
    return ExpEnv(model, dataset, algo_new, label, f"outputs/layer")


if __name__ == "__main__":
    # params1 = list(
    #     itertools.product(
    #         ModelLLaMA3.get_layers_scattered(4),
    #         [EasyEditAlgo("ROME"), EasyEditAlgo("FT-M")],
    #         DatasetsMQCF2hop200 + DatasetsMQCF3hop100,
    #         [ModelLLaMA3],
    #     )
    # )
    # params2 = list(
    #     itertools.product(
    #         ModelQwen2p5.get_layers_scattered(4),
    #         [EasyEditAlgo("ROME"), EasyEditAlgo("FT-M")],
    #         DatasetsMQCF2hop200 + DatasetsMQCF3hop100,
    #         [ModelQwen2p5],
    #     )
    # )
    params2 = list(
        itertools.product(
            ModelQwen2p5.get_layers_scattered(4),
            [EasyEditAlgo("ROME")],
            DatasetsMQCF2hop200[:1],
            [ModelQwen2p5],
        )
    )
    envs = list(itertools.starmap(env_maker, params2))
    lazyexp.run_exps("layers", envs, DEVICE_FREE, cmd_maker)
