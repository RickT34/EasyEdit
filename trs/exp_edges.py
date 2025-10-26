import lazyexp
from env import *
from utils import make_cmd_maker

DEVICE_FREE = [3, 5, 6, 7]

cmd_maker = make_cmd_maker(lambda _, __: ["--only_pre"])
K = 4

if __name__ == "__main__":
    envs = []
    for model in [ModelLLaMA3, ModelQwen2p5]:
        for i in range(K):
            dataset = dataclasses.replace(DatasetMQCFAllEdges, range=f"{i+1}q{K}")
            envs.append(ExpEnv(model, dataset, EasyEditAlgo("ROME"), "allEdges", "outptus/allEdges"))
        
    lazyexp.run_exps("allEdges", envs, DEVICE_FREE, cmd_maker)
