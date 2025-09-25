import env
import lazycleaner
import itertools
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
ALGOs = [
    'FT-M'
    ]
# XX = list(range(0, 32, 4))+[31]
XX = list(range(0, 28, 4)) + [27]
LABELs = list(f"layer{i}" for i in XX)

N = 3
DSS = list(f'mq_cf_sample100_{N}hop{i+1}' for i in range(N)) 

MODEL_NAMES = [
    'llama3-8b',
    'qwen2.5-7b'
]
def metric_nhop(*case):
    return sum(case[:3])/3

def metric_es(loc: int):
    def metric(*case):
        return case[3+loc]
    return metric
def metric_filter(loc:int):
    def metric(*case):
        return all(case[i]>0 for i in range(3, len(case)) if i!=3+loc)
    return metric


def summery(model_name):

    table_nhop = defaultdict(dict)
    table_sample = defaultdict(dict)
    for i, ds in enumerate(DSS):
        data_count = {}
        for algo, (j, label) in itertools.product(ALGOs, enumerate(LABELs)):
            expenv = env.ExpEnv(model_name, ds, algo, label, 'All', "outputs_layer")
            data = lazycleaner.LazyCleaner(expenv, dataset_dir=f'dataset/{ds[:-1]}')
            metrics_nhop = data.metrics_samplewise(metric_nhop)
            metrics_sample = data.metrics_samplewise(metric_es(i))
            metrics_filter = data.metrics_samplewise(metric_filter(i))
            data_count[ds] = metrics_filter[:, 0].sum()
            colname = f"{ds}+{algo}"
            table_nhop[label][colname] = metrics_nhop[metrics_filter[:, 0], 1].mean()
            table_sample[label][colname] = metrics_sample[metrics_filter[:, 0], 1].mean()
            
            
        print(data_count)
    df = pd.DataFrame(table_nhop)
    print(df)
    df_sample = pd.DataFrame(table_sample)
    
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    COLORS = ['#fe4365', '#00a8c6', '#64DD17']
    for ds, color in zip(DSS, COLORS):
        colname = f"{ds}+FT-M"
        plt.plot(XX, df.loc[colname], label=f"$Nhop_{ds[-1]}$", marker='o', color=color)
        plt.plot(XX, df_sample.loc[colname], label=f"$Edit_{ds[-1]}$", marker='.', color=color, linestyle='--', lw=1, alpha=0.5)
    plt.legend()
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Layer-wise Accuracy")
    plt.xlim((-0.5,31.5))
    plt.xticks(XX)
    plt.savefig('data_figs/fig_layer.png')
        # df_sample = pd.DataFrame(table_sample)
        # print(df_sample)
            
summery(MODEL_NAMES[1])