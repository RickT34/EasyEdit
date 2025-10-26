import itertools
import lazyexp
from env import *
from utils import make_cmd_maker

DEVICE_FREE = [3, 5, 6, 7]

cmd_maker = make_cmd_maker()


def env_maker(algo: AlgoEnv, dataset: DatasetEnv, model: ModelEnv):
    return ExpEnv(model, dataset, algo, 'nhop', f"outputs/nhop")


if __name__ == "__main__":
    params1 = list(
        itertools.product(
            [EasyEditAlgo(x) for x in ["AlphaEdit", "FT-M", "LoRA", "MEMIT", "QLoRA", "ROME", "UltraEdit"]],
            DatasetsMQCF2hop800,
            [ModelLLaMA3, ModelQwen2p5],
        )
    )
    envs = list(itertools.starmap(env_maker, params1))
    lazyexp.run_exps('nhop',envs, DEVICE_FREE, cmd_maker)
