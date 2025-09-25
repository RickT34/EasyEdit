import env
import lazycleaner
import itertools
import pandas as pd
from collections import defaultdict
ALGOs = [
    'AlphaEdit', 
    'LoRA', 
    'ROME'
    ]
LABELs = [
    'mq_cf_sample800_2hop1', 
    'mq_cf_sample800_2hop2'
    ]
def metric_nhop(*case):
    return sum(case[:3])/3

def metric_es(loc: int):
    def metric(*case):
        return case[3+loc]
    return metric

def metric_filter(loc:int):
    def metric(*case):
        return case[4-loc] > 0
    return metric


def summery():

    table_nhop = defaultdict(dict)
    table_sample = defaultdict(dict)

    for algo, (i, label) in itertools.product(ALGOs, enumerate(LABELs)):
        expenv = env.ExpEnv('llama3-8b', label, algo, label, 'All')
        data = lazycleaner.LazyCleaner(expenv, 'outputs', use_cache=False)
        metrics_nhop = data.metrics_samplewise(metric_nhop)
        metrics_sample = data.metrics_samplewise(metric_es(i))
        metrics_filter = data.metrics_samplewise(metric_filter(i))
        print(label, metrics_filter[:, 0].sum())
        table_nhop[label][algo] = metrics_nhop[metrics_filter[:, 0], 1].mean()
        table_sample[label][algo] = metrics_sample[metrics_filter[:, 0], 1].mean()
        
    df = pd.DataFrame(table_nhop)
    print(df)

    df_sample = pd.DataFrame(table_sample)
    print(df_sample)

def spec1():
    algo = 'AlphaEdit'
    label = LABELs[0]
    expenv = env.ExpEnv('llama3-8b', label, algo, label, 'All')
    data0 = lazycleaner.LazyCleaner(expenv, 'outputs', use_cache=False, judge_func=lazycleaner.judge_isin)
    data1 = lazycleaner.LazyCleaner(expenv, 'outputs', use_cache=False, judge_func=lazycleaner.judge_isinor)
    
    metrices0 = data0.metrics_samplewise(metric_es(0))
    metrices1 = data1.metrics_samplewise(metric_es(0))
    
    diff = metrices0[:, 1] != metrices1[:, 1]
    print(diff.sum())
    labels = {3:'edit'}
    for i in diff.nonzero()[0]:
        data0.display_sample(i, labels)
        data1.display_sample(i, labels)
        input()
        

if __name__ == '__main__':
    summery()