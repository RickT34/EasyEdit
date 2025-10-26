import env
import lazycleaner
from pathlib import Path
import sqlite3

AXISES_ENV = {
    "Model": lambda x: x.model.name,
    "Algo": lambda x: x.algo.name,
    "Dataset": lambda x: x.dataset.name,
    "Label": lambda x: x.label,
}


def compute_filter(data: lazycleaner.LazyCleaner):
    return data.metrics_samplewise(
        lambda *case: case[0] >0 and case[2]>0
    )


def compute_nhop12(data: lazycleaner.LazyCleaner):
    nhop = data.metrics_samplewise(lambda *case: sum(case[3:6]) / 3)
    filter = compute_filter(data)
    return nhop[filter[:, 0], 1].mean().item()

def compute_nhop23(data: lazycleaner.LazyCleaner):
    nhop = data.metrics_samplewise(lambda *case: sum(case[6:9]) / 3)
    filter = compute_filter(data)
    return nhop[filter[:, 0], 1].mean().item()

def compute_edit(data: lazycleaner.LazyCleaner):
    edit = data.metrics_samplewise(
        lambda *case: case[1]
    )
    filter = compute_filter(data)
    return edit[filter[:, 0], 1].mean().item()


AXISES_VAL = {
    "Nhop12": compute_nhop12,
    "Nhop23": compute_nhop23,
    "Edit": compute_edit,
    "Count": lambda data: compute_filter(data)[:, 0].sum().item(),
}

DATA_CLEANED_DIR = Path("data/data_cleaned")


def create_db(name: str):

    conn = sqlite3.connect(f"{DATA_CLEANED_DIR/name}.db")
    c = conn.cursor()

    sql = (
        "CREATE TABLE data ("
        + ", ".join(f"{k} TEXT" for k in AXISES_ENV.keys())
        + ", "
        + ", ".join(f"{k} REAL" for k in AXISES_VAL.keys())
        + ") strict;"
    )
    c.execute(sql)

    conn.commit()

    def _insert_data(k: tuple):
        c.execute(f"INSERT INTO data VALUES ({', '.join('?' for _ in k)})", k)
        conn.commit()

    return _insert_data


def summery(name: str):

    envs = env.loadEnvs(name)
    db_insert = create_db(name)
    for expenv in envs:
        data = lazycleaner.LazyCleaner(expenv)
        k1 = (ax(expenv) for ax in AXISES_ENV.values())
        k2 = (ax(data) for ax in AXISES_VAL.values())
        k = (*k1, *k2)
        db_insert(k)


summery("layersC")
