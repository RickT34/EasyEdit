import itertools
import lazyexp
from env import *
from utils import make_cmd_maker

DEVICE_FREE = [3, 5, 6, 7]

cmd_maker = make_cmd_maker()


def env_maker(algo: AlgoEnv, dataset: DatasetEnv, model: ModelEnv):
    return ExpEnv(model, dataset, algo, 'dynamic', f"outputs/dynamic", tags={"trace_weight_change":True})


if __name__ == "__main__":
    params1 = list(
        itertools.product(
            [EasyEditAlgo(x) for x in ["FT-M"]],
            DatasetsMQCF2hop800,
            [ModelLLaMA3, ModelQwen2p5],
        )
    )
    envs = list(itertools.starmap(env_maker, params1))
    lazyexp.run_exps('nhop',envs, DEVICE_FREE, cmd_maker)
