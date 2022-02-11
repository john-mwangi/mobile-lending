# %% [markdown]
# # Objectives
# Determine if a customer will default on a mobile loan.
#
# # Process
# * Merge datasets
#   * Summarise historic loans
#   * Check duplicates
#   * Merge data
# * EDA
#   * Fix data types
#   * Independent variable
#   * Dependent vars: nulls impute*
#   * Numeric vars: mean median, range, scaling
#   * Categorical vars: unique values
#   * Dates: ages, month of year, day of month
#   * Coordinates: administrative regions
# * Encoding
#   * Label
#   * One hot
# * SMOTE
# * Base models
#   * Model selection
#   * CV, hyper-parameter tuning
#   * Fit, predict probabilities
# * Model stacking
#   * Fit
#   * Predict (PD)
# * Model evaluation
#   * Confusion matrix, ROC AUC, balanced accuracy
# * Optimisation
#   * Operating point
#   * Adjusted threshold
#   * Re-evaluation

# %% [markdown]
# # Packages

# %%
import pandas as pd
import numpy as np
import dill

from matplotlib import pyplot as plt

from arcgis.gis import GIS
from arcgis.geocoding import reverse_geocode

from datetime import datetime
from smote_variants import MWMOTE

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    auc,
)

# %% [markdown]
# # Data importation
# Data source: https://zindi.africa/competitions/data-science-nigeria-challenge-1-loan-default-prediction/data
#
# **Dataset description:**
# * traindemographics.csv: demographic information of loan borrowers
# * trainperf.csv: This is the repeat loan that the customer has taken for which we need to predict the performance of. Basically, we need to predict whether this loan would default given all previous loans and demographics of a customer. good_bad_flag (good = settled loan on time; bad = did not settled loan on time) - this is the target variable that we need to predict
# * trainprevloans.csv: This dataset contains all previous loans that the customer had prior to the loan above that we want to predict the performance of. Each loan will have a different systemloanid, but the same customerid for each customer.
#

# %%
traindemographics_raw = pd.read_csv("../inputs/train/traindemographics.csv")
trainperf_raw = pd.read_csv("../inputs/train/trainperf.csv")
trainprevloans_raw = pd.read_csv("../inputs/train/trainprevloans.csv")

testdemographics_raw = pd.read_csv("../inputs/test/testdemographics.csv")
testperf_raw = pd.read_csv("../inputs/test/testperf.csv")
testprevloans_raw = pd.read_csv("../inputs/test/testprevloans.csv")

# %%
traindemographics_raw.shape

# %%
trainperf_raw.shape

# %%
trainprevloans_raw.shape

# %%
traindemographics_proc = traindemographics_raw.copy()
trainperf_proc = trainperf_raw.copy()
trainprevloans_proc = trainprevloans_raw.copy()

testdemographics_proc = testdemographics_raw.copy()
testperf_proc = testperf_raw.copy()
testprevloans_proc = testprevloans_raw.copy()

# %% [markdown]
# # Merge datasets
# ## Summarise previous loans dataset
# ### Fix data types

# %%
trainprevloans_proc.customerid.sample(n=1)
trainprevloans_proc[
    trainprevloans_proc.customerid == "8a2a81a74ce8c05d014cfb32a0da1049"
].sort_values(by="approveddate", ascending=True)

# %%
trainprevloans_proc.dtypes

# %%
def convert_dates(df):
    """
    Converts date columns to datetime.
    :param df: dataframe of raw data
    """
    date_cols = df.columns[df.columns.str.contains("date")]
    date_df = df[date_cols].transform(func=pd.to_datetime)
    trim_df = df.drop(columns=date_cols)
    new_df = pd.concat(objs=[trim_df, date_df], axis=1)
    return new_df


# %%
trainprevloans_proc = convert_dates(trainprevloans_proc)
testprevloans_proc = convert_dates(testprevloans_proc)
trainprevloans_proc.dtypes

# %% [markdown]
# ### Summarise numeric columns

# %%
def summ_numeric(df):
    """
    Summarises key numeric information.
    :param df: The output of convert_dates function.
    """
    df_summ = df.groupby(by="customerid", as_index=False).aggregate(
        loans_taken=("loannumber", "max"),
        avg_loanamount=("loanamount", "mean"),
        avg_term=("termdays", "mean"),
    )

    return df_summ


# %%
trainprevloans_summ = summ_numeric(df=trainprevloans_proc)
testprevloans_summ = summ_numeric(df=testprevloans_proc)
trainprevloans_summ.shape

# %%
trainprevloans_summ.head()

# %% [markdown]
# ### Summarise date columns
# Date transformations:
# * approveddate: none, creationdate will be used
# * creationdate: does the month of year and day of month affect repayment?
# * closeddate: none, reflected in default
# * firstduedate: none
# * firstrepaiddate: how soon does the customer pay after loan approval (sign of stress?)

# %%
def summ_dates(df):
    """
    Converts certain dates to their numeric equivalents and calculates daily
    interest rates.
    :param df: The output of summ_numeric function.
    """
    df["creation_month"] = df.creationdate.dt.month
    df["creation_day"] = df.creationdate.dt.day

    firstrepaid_days = df.firstrepaiddate - df.approveddate
    firstrepaid_days = firstrepaid_days.apply(lambda x: x.days)
    df["firstrepaid_days"] = firstrepaid_days

    interest_days = (df.closeddate - df.approveddate).apply(lambda x: x.days)
    interest_amt = df.totaldue - df.loanamount
    daily_int = interest_amt / interest_days
    daily_int = daily_int.replace({float("inf"): 0})
    df["daily_int"] = daily_int

    return df


# %%
trainprevloans_proc = summ_dates(df=trainprevloans_proc)
testprevloans_proc = summ_dates(df=testprevloans_proc)
trainprevloans_proc.shape

# %%
trainprevloans_proc.head()

# %%
def group_days(prev_loans, loans_summ):
    """
    Summarises information from summ_dates function.
    :param prev_loans: The output of summ_dates function
    :param loans_summ: The output of summ_numeric function
    """
    new_summ = (
        prev_loans.groupby(by="customerid", as_index=False)
        .aggregate(
            avg_creation_month=("creation_month", "mean"),
            avg_creation_day=("creation_day", "mean"),
            avg_firstrepaid_days=("firstrepaid_days", "mean"),
            avg_daily_int=("daily_int", "mean"),
        )
        .merge(right=loans_summ, how="left", on="customerid")
    )

    return new_summ


# %%
trainprevloans_summ = group_days(
    prev_loans=trainprevloans_proc, loans_summ=trainprevloans_summ
)
testprevloans_summ = group_days(
    prev_loans=testprevloans_proc, loans_summ=testprevloans_summ
)
trainprevloans_summ.shape

# %%
trainprevloans_summ.head()

# %% [markdown]
# ## Merge all datasets

# %%
def merge_datasets(perf_df, demo_df, summ_df):
    """
    Merges all datasets.
    :param perf_df: trainperf or equivalent
    :param demo_df: traindemographics or equivalent
    :param summ_df: The output of group_days function
    """
    perf_df = perf_df.drop_duplicates(subset=["customerid"])
    demo_df = demo_df.drop_duplicates(subset=["customerid"])

    merge_1 = pd.merge(
        left=perf_df, right=demo_df, how="left", on="customerid"
    )
    merged_df = pd.merge(
        left=merge_1, right=summ_df, how="left", on="customerid"
    )

    return merged_df


# %%
train_merged_raw = merge_datasets(
    perf_df=trainperf_proc,
    demo_df=traindemographics_proc,
    summ_df=trainprevloans_summ,
)
test_merged_raw = merge_datasets(
    perf_df=testperf_proc,
    demo_df=testdemographics_proc,
    summ_df=testprevloans_summ,
)
train_merged_raw.shape

# %%
train_merged_proc = train_merged_raw.copy()
test_merged_proc = test_merged_raw.copy()

# %% [markdown]
# # EDA
# ## Fix data types

# %%
train_merged_proc.head()

# %%
train_merged_proc.dtypes
test_merged_proc.dtypes

# %%
test_merged_proc.columns[test_merged_proc.columns.str.contains("date")]
test_merged_proc[
    test_merged_proc.columns[test_merged_proc.columns.str.contains("date")]
]

# %%
train_merged_proc = convert_dates(train_merged_proc)
# test_merged_proc = convert_dates(test_merged_proc)

# %% [markdown]
# ## Dependent variable

# %%
train_merged_proc.good_bad_flag.isnull().sum()

# %% [markdown]
# * Synthetic data generation will be required to address the class imbalance.
# * `Good` will be encoded as 1, `Bad` will be encoded as 0

# %%
train_merged_proc.good_bad_flag.value_counts() / len(
    train_merged_proc.good_bad_flag
)

# %%
train_merged_proc["good_bad_flag"] = train_merged_proc[
    "good_bad_flag"
].replace({"Good": 1, "Bad": 0})

train_merged_proc.good_bad_flag.value_counts()

# %% [markdown]
# ## Null values
# Drop variables with >20% missingness and impute the rest.

# %%
train_merged_proc.isnull().sum() / train_merged_proc.shape[0]

# %%
train_merged_proc.isnull().sum() / train_merged_proc.shape[0] > 0.3
null_cols = train_merged_proc.columns[
    train_merged_proc.isnull().sum() / train_merged_proc.shape[0] > 0.3
]
train_merged_proc = train_merged_proc.drop(columns=null_cols)
train_merged_proc.shape

# %%
np.intersect1d(ar1=train_merged_proc.columns, ar2=test_merged_proc.columns)

test_merged_proc = test_merged_proc[
    np.intersect1d(ar1=train_merged_proc.columns, ar2=test_merged_proc.columns)
]
test_merged_proc.shape

# %%
imputer = SimpleImputer(
    strategy="most_frequent"
)  # TODO: replace with RandomForest

train_imp = train_merged_proc.drop(columns="good_bad_flag")
imputer.fit(X=train_imp)

# %%
test_merged_proc = pd.DataFrame(
    imputer.transform(X=test_merged_proc[imputer.feature_names_in_]),
    columns=imputer.feature_names_in_,
)

train_merged_proc = pd.DataFrame(
    imputer.transform(X=train_imp), columns=train_imp.columns
).assign(good_bad_flag=train_merged_proc.good_bad_flag)

train_merged_proc.shape

# %%
imputer.feature_names_in_
imputer.statistics_
{k: v for k, v in zip(imputer.feature_names_in_, imputer.statistics_)}

# %% [markdown]
# ## Numeric variables
# Check mean, median, and range of values.
# ### Revert data types

# %%
train_merged_proc.dtypes

# %%
# "most_frequent" imputation changes numeric variables to object


def revert_numeric(proc, raw):
    num_cols = np.intersect1d(
        ar1=raw.select_dtypes(include=np.number).columns, ar2=proc.columns
    )

    num_repl = {k: v for k, v in zip(num_cols, ["float"] * len(num_cols))}
    proc = proc.astype(dtype=num_repl)
    return proc


# %%
train_merged_proc = revert_numeric(
    proc=train_merged_proc, raw=train_merged_raw
)
test_merged_proc = revert_numeric(proc=test_merged_proc, raw=test_merged_raw)
train_merged_proc.dtypes

# %% [markdown]
# ### Summary stats

# %% [markdown]
# Scaling will be applied due to the wide range i.e. max value is more than 1 sd from the mean. This will be applied on the amounts.

# %%
train_merged_proc.describe().transpose()

# %%
scaler = StandardScaler()

sc_cols = [
    "loanamount",
    "totaldue",
    "avg_daily_int",
    "avg_loanamount",
    "loans_taken",
]

train_merged_proc = pd.concat(
    objs=[
        train_merged_proc.drop(columns=sc_cols),
        pd.DataFrame(
            scaler.fit_transform(X=train_merged_proc[sc_cols]), columns=sc_cols
        ),
    ],
    axis=1,
)

train_merged_proc.shape

# %%
def scale_data(testdata, scaler, sc_cols):
    """
    Scales test data.
    :param testdata: dataframe containing test data
    :param scaler: a fitted scaler
    :param sc_cols: a list of columns to scale
    """

    testdata_sc = pd.concat(
        objs=[
            testdata.drop(columns=sc_cols),
            pd.DataFrame(
                scaler.fit_transform(X=testdata[sc_cols]), columns=sc_cols
            ),
        ],
        axis=1,
    )

    return testdata_sc


# %%
test_merged_proc = scale_data(
    testdata=test_merged_proc, scaler=scaler, sc_cols=sc_cols
)
test_merged_proc.shape

# %% [markdown]
# ## Categorical variables
# Check number of distinct values.

# %%
train_merged_proc.select_dtypes(include=pd.Categorical).nunique()

# %% [markdown]
# * [Tier 1 banks](https://thenationonlineng.net/tier-1-banks-assets-hit-n46tr/#:~:text=The%20banks%20include%3A%20United%20Bank,of%20Nigeria%20Holdings%20(FBNH).)
# * [Tier 2 banks](https://businessday.ng/banking/article/tier-2-banks-maintains-npl-ratios-lower-than-5-0-threshold/#:~:text=Banks%20that%20fall%20under%20the,Plc%20(3.5%25)%20NPL%20ratios.)

# %%
train_merged_proc.bank_name_clients.value_counts()  # TODO: replace with bank tiers
# train_merged_proc.bank_name_clients.value_counts().to_csv("../outputs/banks.csv")

# %% [markdown]
# ## Date variables
# Change date columns to numeric.

# %%
train_merged_proc.select_dtypes(include="datetime64").columns

# %%
def customer_age(df):
    """
    Calculates customer age in days and removes remaining datetime variables
    :param df: The output of merge_datasets function
    """
    df["birthdate"] = pd.to_datetime(df.birthdate)

    customer_age = df.birthdate.apply(lambda x: datetime.now() - x).apply(
        lambda x: x.days
    )

    df["customer_age"] = customer_age

    date_cols = df.columns[df.columns.str.contains("date")]
    df = df.drop(columns=date_cols)

    return df


# %%
train_merged_proc = customer_age(df=train_merged_proc)
test_merged_proc = customer_age(df=test_merged_proc)
train_merged_proc.shape

# %%
train_merged_proc.head()

# %% [markdown]
# ## Coordinates

# %%
# TODO: use administrative regions

# gis = GIS()
# test_geo = reverse_geocode([2.2945, 48.8583])
# test_geo.get("address").get("Region")

# test_geo = traindemographics_proc.head(3)
# loc_res = {}

# for cust, long, lat in zip(test_geo.customerid, test_geo.longitude_gps, test_geo.latitude_gps):
#     try:
#         loc = reverse_geocode([long, lat])
#         loc_res[cust]=loc
#     except Exception as e:
#         loc_res[cust]=e

# loc_res

# ((traindemographics_proc.shape[0]/3)*0.9)/60

# %% [markdown]
# ## Identifier vars

# %%
train_merged_proc.columns.str.endswith("id")
id_vars = train_merged_proc.columns[
    train_merged_proc.columns.str.endswith("id")
]
id_vars

# %%
train_merged_proc = train_merged_proc.drop(columns=id_vars)
test_merged_proc = test_merged_proc.drop(columns=id_vars)
train_merged_proc.shape

# %%
test_merged_proc.shape

# %%
test_merged_proc.isna().sum()

# %% [markdown]
# # Encoding
# We shall do one-hot encoding.

# %%
train_merged_proc.select_dtypes(include=object).head()

# %%
train_merged_proc_y = train_merged_proc.good_bad_flag
train_merged_proc_X = train_merged_proc.drop(columns=["good_bad_flag"])

# %% [markdown]
# ## Dummer
# This is a helper dataframe for ensuring that we get consistent dummification results. Especially useful for handling single responses.

# %%
train_merged_proc_X.select_dtypes(include=object)

dummer_dict = dict()
for col in train_merged_proc_X.select_dtypes(include=object).columns:
    dummer_dict[col] = pd.unique(train_merged_proc_X[col])

dummer_dict

# %%
dummer_df = pd.DataFrame.from_dict(dummer_dict, orient="index").transpose()
dummer_df

# %% [markdown]
# ## Dummies

# %%
train_merged_proc_X = pd.get_dummies(train_merged_proc_X, drop_first=False)
train_merged_proc_X.shape

# %%
test_dumm_df = pd.concat(objs=[test_merged_proc, dummer_df], axis=0)
test_merged_proc_X = pd.get_dummies(test_dumm_df, drop_first=False)
test_merged_proc_X = test_merged_proc_X.iloc[: test_merged_proc.shape[0], :]
test_merged_proc_X.shape

# %% [markdown]
# # MWMOTE

# %%
oversampler = MWMOTE(random_state=123, n_jobs=-1)

# %%
np.random.seed(123)

train_sm_X, train_sm_y = oversampler.sample(
    X=np.asarray(train_merged_proc_X), y=train_merged_proc_y
)

train_sm_X.shape

# %%
np.unique(train_sm_y, return_counts=True)

# %%
train_sm_X = pd.DataFrame(train_sm_X, columns=train_merged_proc_X.columns)
train_sm_y = train_sm_y.tolist()

# %% [markdown]
# # Train test split

# %%
train_X, test_X, train_y, test_y = train_test_split(
    train_sm_X,
    train_sm_y,
    train_size=0.7,
    shuffle=True,
    random_state=123,
    stratify=train_sm_y,
)

# %% [markdown]
# # Base models
# ## Model selection

# %%
dt = DecisionTreeClassifier(random_state=123)
rf = RandomForestClassifier(random_state=123)
gb = GradientBoostingClassifier(random_state=123)
svm = SVC(random_state=123)
knn = KNeighborsClassifier()
nn = MLPClassifier(random_state=123)

# %% [markdown]
# ## Cross validation

# %%
kf = StratifiedKFold(n_splits=10)

# %% [markdown]
# ## Hyper-parameter tuning
# Random search tuning on selected parameters.

# %%
dt_params = {
    "max_depth": np.arange(1, 5),
    "min_samples_leaf": np.arange(2, 10, 2),
    "max_features": ["auto"],
    "max_leaf_nodes": np.arange(1, 10),
}

rf_params = {
    "max_depth": np.arange(3, 7),
    "max_samples": np.arange(2, 9),
    "max_leaf_nodes": np.arange(1, 10),
}

gb_params = {
    "n_estimators": [50, 100, 150],
    "min_samples_leaf": np.arange(1, 10),
    "max_depth": np.arange(3, 7),
}

knn_params = {"n_neighbors": [3, 5, 7, 10], "leaf_size": [10, 20, 30]}

svm_params = {
    "C": [1.0, 2.0, 3.0, 4.0, 5.0],
    "max_iter": [5, 10],
    "probability": [True],
}

nn_params = {
    "random_state": [123],
    "early_stopping": [True],
    "max_iter": [100],
}

# %%
dt_cv = RandomizedSearchCV(
    estimator=dt,
    param_distributions=dt_params,
    scoring="roc_auc",
    n_jobs=-1,
    cv=kf,
    random_state=123,
    return_train_score=True,
)
rf_cv = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_params,
    scoring="roc_auc",
    n_jobs=-1,
    cv=kf,
    random_state=123,
    return_train_score=True,
)
gb_cv = RandomizedSearchCV(
    estimator=gb,
    param_distributions=gb_params,
    scoring="roc_auc",
    n_jobs=-1,
    cv=kf,
    random_state=123,
    return_train_score=True,
)
knn_cv = RandomizedSearchCV(
    estimator=knn,
    param_distributions=knn_params,
    scoring="roc_auc",
    n_jobs=-1,
    cv=kf,
    random_state=123,
    return_train_score=True,
)
svm_cv = RandomizedSearchCV(
    estimator=svm,
    param_distributions=svm_params,
    scoring="roc_auc",
    n_jobs=-1,
    cv=kf,
    random_state=123,
    return_train_score=True,
)
nn_cv = RandomizedSearchCV(
    estimator=nn,
    param_distributions=nn_params,
    scoring="roc_auc",
    n_jobs=-1,
    cv=kf,
    random_state=123,
    return_train_score=True,
)

# %% [markdown]
# ## Fit models

# %%
dt_cv.fit(X=train_X, y=train_y)
rf_cv.fit(X=train_X, y=train_y)
gb_cv.fit(X=train_X, y=train_y)
svm_cv.fit(X=train_X, y=train_y)
knn_cv.fit(X=train_X, y=train_y)

# %% [markdown]
# ## Predict
# Probability of default from base models to generate training data for ensemble model.

# %%
dt_probs_tr = dt_cv.predict_proba(train_X)[:, 0]
rf_probs_tr = rf_cv.predict_proba(train_X)[:, 0]
gb_probs_tr = gb_cv.predict_proba(train_X)[:, 0]
svm_probs_tr = svm_cv.predict_proba(train_X)[:, 0]
knn_probs_tr = knn_cv.predict_proba(train_X)[:, 0]

# %%
def base_probs(base_probs):
    """
    Convert base probabilities to dataframe.
    :param base_probs: list of base probabilities in the order below \n
    ["dt_probs", "rf_probs", "gb_probs", "svm_probs", "knn_probs"]
    """
    base_df = pd.DataFrame()

    for p in base_probs:
        temp_df = pd.DataFrame(p)
        base_df = pd.concat(objs=[base_df, temp_df], axis=1)

    base_df.columns = [
        "dt_probs",
        "rf_probs",
        "gb_probs",
        "svm_probs",
        "knn_probs",
    ]

    return base_df


# %%
base_probs_tr = base_probs(
    base_probs=[
        dt_probs_tr,
        rf_probs_tr,
        gb_probs_tr,
        svm_probs_tr,
        knn_probs_tr,
    ]
)
base_probs_tr.shape

# %%
base_probs_tr.head()

# %% [markdown]
# # Stacked model
# ## Fit model

# %%
nn_cv.fit(X=base_probs_tr, y=train_y)

# %% [markdown]
# ## Predit
# These represent the probabilities of defaults of the test set.

# %%
dt_probs_te = dt_cv.predict_proba(test_X)[:, 0]
rf_probs_te = rf_cv.predict_proba(test_X)[:, 0]
gb_probs_te = gb_cv.predict_proba(test_X)[:, 0]
svm_probs_te = svm_cv.predict_proba(test_X)[:, 0]
knn_probs_te = knn_cv.predict_proba(test_X)[:, 0]

# %%
base_probs_te = base_probs(
    base_probs=[
        dt_probs_te,
        rf_probs_te,
        gb_probs_te,
        svm_probs_te,
        knn_probs_te,
    ]
)
base_probs_te.shape

# %%
default_probs = nn_cv.predict_proba(base_probs_te)[:, 0]
nn_preds = nn_cv.predict(base_probs_te)
nn_preds

# %% [markdown]
# # Evaluate model

# %%
cm_res = confusion_matrix(y_true=test_y, y_pred=nn_preds)
cm_res

# %%
np.unique(test_y, return_counts=True)

# %%
class_0 = cm_res[0, 0] / (cm_res[0, 0] + cm_res[0, 1])
class_1 = cm_res[1, 1] / (cm_res[1, 1] + cm_res[1, 0])

print(f"Accuracy in predicting 0 [default]: { class_0 }")
print(f"Accuracy in predicting 1 [no default]: { class_1 }")

# %%
balanced_accuracy_score(y_true=test_y, y_pred=nn_preds)

# %%
roc_auc_score(y_true=test_y, y_score=nn_preds)

# %% [markdown]
# # Optimisation
# ## Operating point
# We want to determine the point in the curve that's closest to the top-left part of the graph.

# %%
fpr, tpr, thresholds = roc_curve(y_true=test_y, y_score=nn_preds)
auc(x=fpr, y=tpr)

# %%
def operating_point(x, y, points):
    """
    Determine the operating point of a curve. \n
    Args
    ----
    x: An array e.g., FPR
    y: An array e.g., TPR
    point: a list representing terminal coordinates [x,y]

    Returns
    ----
    Dataframe with distances
    """
    dist = ((points[0] - x) ** 2 + (points[1] - y) ** 2) ** 0.5
    res = list(zip(x, y, dist))
    res_df = pd.DataFrame(res, columns=["x", "y", "dist"])
    res_df = res_df.sort_values(by="dist").reset_index(drop=True)
    return res_df


# %%
op = operating_point(x=fpr, y=tpr, points=[0, 1])
op

# %%
plt.plot(
    fpr, tpr, color="darkorange", label=f"AUC={round(auc(x=fpr, y=tpr),3)}"
)

# plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.axvline(x=op["x"][0], color="green", linestyle="--", label="Adjusted PD")
plt.axhline(y=op["y"][0], color="green", linestyle="--")
plt.axvline(x=0.5, color="navy", linestyle="--", label="Default PD")
plt.ylabel("True Positive Rate")
plt.title("Mobile Lending ROC Curve")
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ## Adjusted threshold
# This is the new PD cut-off (**0.2048780487804878**). If PD is greater than this, the customer is flagged as likely to default.

# %%
op_thr = op["x"][0]
op_thr

# %%
adjusted_preds = [0 if default > (op_thr) else 1 for default in default_probs]
adjusted_preds[:20]

# %% [markdown]
# ## Evaluation
# This has improved the accuracy of detecting defaults from 79% to 84%

# %%
balanced_accuracy_score(y_true=test_y, y_pred=adjusted_preds)

# %%
cm_adj = confusion_matrix(y_true=test_y, y_pred=adjusted_preds)
cm_adj

# %%
class_0 = cm_adj[0, 0] / (cm_adj[0, 0] + cm_adj[0, 1])
class_1 = cm_adj[1, 1] / (cm_adj[1, 1] + cm_adj[1, 0])

print(f"Accuracy in predicting 0 [default]: { class_0 }")
print(f"Accuracy in predicting 1 [no default]: { class_1 }")

# %% [markdown]
# # Export

# %%
api_data = {
    "train_X": train_X,
    "scaler": scaler,
    "dummer_df": dummer_df,
    "null_cols": null_cols,
    "scale_cols": sc_cols,
    "op_thr": op_thr,
    "imputer": imputer,
}

api_models = {
    "dt_cv": dt_cv,
    "rf_cv": rf_cv,
    "gb_cv": gb_cv,
    "svm_cv": svm_cv,
    "knn_cv": knn_cv,
    "nn_cv": nn_cv,
}

# %%
with open(file="../outputs/api_data.pkl", mode="wb") as f:
    dill.dump(obj=api_data, file=f)

with open(file="../outputs/api_models.pkl", mode="wb") as f:
    dill.dump(obj=api_models, file=f)

# %%
ts = datetime.now()
tm = datetime.strftime(ts, "%d_%b")

dill.dump_session(filename=f"../outputs/{tm}.pkl")

# %%
# import dill
# dill.load_session("../outputs/08_Feb.pkl")
