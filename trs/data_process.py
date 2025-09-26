import env
import lazycleaner
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

AXISES = {
    "Model": lambda x: x.model_name,
    "Algo": lambda x: x.algo_name,
    "Dataset": lambda x: x.dataset.name,
    "Label": lambda x: x.label,
}

KEYs = tuple(AXISES.keys())


def summery(envs: list[env.ExpEnv], xylabel: tuple[str, str] ):

    table_nhop = defaultdict(dict)
    table_sample = defaultdict(dict)
    ds_counts = {}
    for expenv in envs:
        k = tuple(AXISES[i](expenv) for i in xylabel)
        data = lazycleaner.LazyCleaner(expenv)
        loc = expenv.dataset.tags["loc"]

        metrics_nhop = data.metrics_samplewise(lambda *case: sum(case[:3]) / 3)
        metrics_sample = data.metrics_samplewise(lambda *case: case[3 + loc])
        metrics_filter = data.metrics_samplewise(
            lambda *case: all(x > 0 or i == loc for i, x in enumerate(case[3:]))
        )

        assert k[1] not in table_nhop[k[0]], f"duplicate key: {k}"
        table_nhop[k[0]][k[1]] = metrics_nhop[metrics_filter[:, 0], 1].mean()
        table_sample[k[0]][k[1]] = metrics_sample[metrics_filter[:, 0], 1].mean()

        count = metrics_filter[:, 0].sum().item()
        ds = expenv.dataset.name
        assert count == ds_counts.setdefault(
            ds, count
        ), f"dataset {ds} count not consistent: {count} != {ds_counts[ds]}"

    df_nhop = pd.DataFrame(table_nhop)
    df_sample = pd.DataFrame(table_sample)
    df_count = pd.DataFrame(ds_counts, index=["count"])
    return df_nhop, df_sample, df_count

def plot_layers(envs: list[env.ExpEnv]):
    XLABEL = "Label"
    YLABEL = "Dataset"
    df_nhop, df_sample, counts = summery(envs, (XLABEL, YLABEL))
    # print(df_nhop, df_sample, counts, sep='\n\n')
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    COLORS = ['#fe4365', '#00a8c6', '#64DD17']
    XX = [int(s[5:]) for s in df_nhop.columns.values]
    for ds, color in zip(df_nhop.index, COLORS):
        plt.plot(XX, df_nhop.T[ds], label=f"$Nhop_{ds[-1]}$", marker='o', color=color)
        plt.plot(XX, df_sample.T[ds], label=f"$Edit_{ds[-1]}$", marker='.', color=color, linestyle='--', lw=1, alpha=0.5)
    plt.legend()
    plt.xlabel("Layers")
    plt.ylabel("Accuracy")
    plt.title("Layer-wise Accuracy")
    plt.xlim((XX[0]-0.5,XX[-1]+0.5))
    plt.xticks(XX)
    plt.savefig('data_figs/layer.png')

expenv = [env.loadEnv(f"/Data2/tangrui/EasyEdit/trs/exp_history/exp_2/env_{i}.json") for i in range(1, 6)]
# print(*summery(expenv), sep='\n\n')
plot_layers(expenv)
