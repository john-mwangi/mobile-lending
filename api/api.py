from fastapi import FastAPI, Query
from functools import lru_cache
from .utils import Scorer
import pandas as pd

SAMPLES = 2

app = FastAPI(title="Mobile Lending Fast API")

# To run this API on the terminal...
# uvicorn api.api:app --host 0.0.0.0 --port 12000 --reload
# localhost:12000/docs for Swagger documentation & testing

sc = Scorer()

train_X = sc.load_test_data(items=["train_X"])
train_X = train_X[0]
train_X = train_X.sample(n=SAMPLES)
train_X_json = train_X.to_json()


@app.get(path="/")
def read_root():
    return {"Test": "Fast API ran successfully!!"}


@lru_cache(maxsize=128)
@app.get(path="/predict")
def fetch_predictions(
    resp: str = Query(
        default=train_X_json,
        description="Customer details obtained after data merging and preparation.",
    )
) -> dict:

    train_X = pd.read_json(resp)
    op_thr = sc.load_test_data(items=["op_thr"])

    try:
        default_preds, adjusted_preds = sc.obtain_preds(
            data=train_X.sample(n=SAMPLES), op_thr=op_thr
        )
    except Exception as e:
        default_preds, adjusted_preds = ["An error has occured"] * 2

    return {
        "Train X": train_X.shape,
        "Default preds": default_preds,
        "Adjusted preds": adjusted_preds,
        "Threshold": op_thr,
    }
