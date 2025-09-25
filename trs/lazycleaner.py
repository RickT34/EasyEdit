import json

from env import *
import itertools
import dsmerge
from pathlib import Path
import numpy as np
from rich import print

DATASET_DIR = "dataset"


def judge_isin(llm_ans: str, answers: list[str]):
    return 1.0 if any((a.lower() in llm_ans.lower()) for a in answers) else 0.0

def judge_isinor(llm_ans: str, answers: list[str]):
    return 1.0 if any((a.lower() in llm_ans.lower() or llm_ans.lower() in a.lower()) for a in answers) else 0.0

class LazyCleaner:

    def __init__(
        self,
        expenv: ExpEnv,
        use_cache: bool = True,
        dataset_dir: str = DATASET_DIR,
        judge_func=None,
    ):
        self.expenv = expenv
        self.dataset_dir = dataset_dir
        self.judge_func = judge_func or judge_isin

        cache_file = expenv.get_output_path().with_suffix(".cleaned.json")
        if use_cache and cache_file.exists():
            self.cleaned_data = json.load(open(cache_file))
        else:
            self.cleaned_data = self._clean_data()
            json.dump(self.cleaned_data, open(cache_file, "w"), indent=4)

    @property
    def dataset(self):
        if not hasattr(self, "_dataset"):
            self._dataset = json.load(
                open(f"{self.dataset_dir}/{self.expenv.ds_name}.json")
            )
        return self._dataset

    @property
    def rawdata(self):
        if not hasattr(self, "_rawdata"):
            self._rawdata = self.collect_data()
        return self._rawdata

    def collect_data(self):
        output_file = self.expenv.get_output_path()
        data = dsmerge.auto_collect(output_file.parent)
        return data

    def _clean_data(self):
        assert len(self.rawdata) == len(
            self.dataset
        ), f"length of raw data and mq data are not equal: {len(self.rawdata)}!={len(self.dataset)}"
        cleaned_data = ([], [])
        for r, d in zip(self.rawdata, self.dataset):
            for i, s in enumerate(("pre", "post")):
                l_llm_answer = r[s][1]
                l_answers = d["test_answers"]
                assert len(l_llm_answer) == len(
                    l_answers
                ), f"length of llm answer and answers are not equal: {l_llm_answer}"
                result = list(
                        itertools.starmap(self.judge_func, zip(l_llm_answer, l_answers))
                    )
                cleaned_data[i].append(result)
        return cleaned_data

    def metrics_samplewise(self, metric_func):
        return np.array(
            list((metric_func(*i), metric_func(*j)) for i, j in zip(*self.cleaned_data))
        )

    def display_sample(self, idx: int, labels: dict[int, str]):
        case = self.dataset[idx]
        print(f"========== Sample {idx} ==========")
        print(f"Request: {case['prompt']} {case['target_new']}")

        def tranc_sentance(s: str):
            NUMCHARS = 100
            s = s[s.rfind('\n') + 1 :]
            return "..." + s[-NUMCHARS:] if len(s) > NUMCHARS + 3 else s

        def display_qa(q: str, llm_ans: str, answers: list[str]):
            print(f"    -- {tranc_sentance(q)}")
            print(f"    -- LLM: {tranc_sentance(llm_ans)}")
            print(f"    -- Ans: {' | '.join(tranc_sentance(a) for a in answers)}")
            print(f"    -- Judge: {self.judge_func(llm_ans, answers)}")

        for i, s in enumerate(("pre", "post")):
            print(f" --------  {s.upper()}  --------")
            for j, (q, llm_ans, answers) in enumerate(
                zip(case[f"tests"], self.rawdata[idx][s][1], case["test_answers"])
            ):
                if j in labels:
                    print(f'  {j+1}. {labels.get(j, "")}')
                    display_qa(q, llm_ans, answers)
                    print(f"    -- Judge: {self.cleaned_data[i][idx][j]}")
                
