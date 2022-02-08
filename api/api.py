from fastapi import FastAPI
from functools import lru_cache
from .utils import Scorer
import pandas as pd

SAMPLES = 2

app = FastAPI()

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
def fetch_predictions(resp=train_X_json):

    train_X = pd.read_json(resp)
    op_thr = sc.load_test_data(items=["op_thr"])

    default_preds, adjusted_preds = sc.obtain_preds(
        data=train_X.sample(n=SAMPLES), op_thr=op_thr
    )

    return {
        "Loaded models": len(sc.load_models()),
        "Train X": train_X.shape,
        "Default preds": default_preds,
        "Adjusted preds": adjusted_preds,
        "Threshold": op_thr,
    }
