from fastapi import FastAPI
from .utils import Loader
from functools import lru_cache

SAMPLES = 5

app = FastAPI()

# To run this API on the terminal...
# uvicorn api.api:app --host 0.0.0.0 --port 12000 --reload
# localhost:12000/docs for Swagger documentation & testing


@app.get(path="/")
def read_root():
    return {"Test": "Fast API ran successfully!!"}


@app.get(path="/predict")
@lru_cache(maxsize=128)
def fetch_predictions():
    dr = Loader()

    train_X, op_thr = dr.load_test_data()
    default_preds, adjusted_preds = dr.obtain_preds(
        data=train_X.sample(n=SAMPLES), op_thr=op_thr
    )

    return {
        "Loaded models": len(dr.load_models()),
        "Train X": train_X.shape,
        "Default preds": default_preds,
        "Adjusted preds": adjusted_preds,
        "Threshold": op_thr,
    }
