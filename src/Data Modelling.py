# %% [markdown]
# # Objectives
# Determine if a customer will default on a mobile loan.
#
# # Process
# * EDA
#   * Duplicates
#   * Data types
#   * Independent variable
#   * Dependent vars: nulls, impute*
#   * Numeric vars: mean, median, range, scaling
#   * Categorical vars: unique values
#   * Dates: ages, month of year, day of month
#   * Coordinates: administrative regions
# * Merge datasets
#   * Impute*
#   * Drop variables
# * Encoding
#   * Label
#   * One hot
# * SMOTE

# %% [markdown]
# # Packages

# %%
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from arcgis.gis import GIS
from arcgis.geocoding import reverse_geocode

from datetime import datetime

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

# %% [markdown]
# # EDA
# ## Fix data types
# ### Demographic data

# %%
traindemographics_proc.head()

# %%
traindemographics_proc.dtypes

# %%
traindemographics_proc["birthdate"] = pd.to_datetime(
    traindemographics_proc["birthdate"]
)
traindemographics_proc.head()

# %% [markdown]
# ### Previous loans

# %%
trainprevloans_proc.head()

# %%
trainprevloans_proc.dtypes

# %%
trainprevloans_proc.columns.str.contains("date")
trainprevloans_proc.columns[trainprevloans_proc.columns.str.contains("date")]
trainprevloans_dates = trainprevloans_proc[
    trainprevloans_proc.columns[
        trainprevloans_proc.columns.str.contains("date")
    ]
]
trainprevloans_dates = trainprevloans_dates.transform(
    func=lambda x: pd.to_datetime(x)
)
trainprevloans_dates.head()

# %%
trainprevloans_proc.drop(columns=trainprevloans_dates.columns)

trainprevloans_proc = pd.concat(
    objs=[
        trainprevloans_proc.drop(columns=trainprevloans_dates.columns),
        trainprevloans_dates,
    ],
    axis=1,
)

trainprevloans_proc.head()

# %% [markdown]
# ### Current loan

# %%
trainperf_proc.head()

# %%
trainperf_proc.dtypes

# %%
trainperf_proc["approveddate"] = pd.to_datetime(trainperf_proc["approveddate"])
trainperf_proc["creationdate"] = pd.to_datetime(trainperf_proc["creationdate"])
trainperf_proc.head()

# %% [markdown]
# ## Duplicates

# %%
len(trainperf_proc["customerid"]) == len(
    np.unique(trainperf_proc["customerid"])
)
len(trainprevloans_proc["systemloanid"]) == len(
    np.unique(trainprevloans_proc["systemloanid"])
)
len(traindemographics_proc["customerid"]) == len(
    np.unique(traindemographics_proc["customerid"])
)

# %%
traindemographics_proc = traindemographics_proc.drop_duplicates(
    subset=["customerid"]
)

# %% [markdown]
# ## Outcome variable

# %%
trainperf_proc["good_bad_flag"].isnull().sum()

# %% [markdown]
# * Synthetic data generation will be required to address the class imbalance.
# * `Good` will be encoded as 1, `Bad` will be encoded as 0

# %%
trainperf_proc["good_bad_flag"].value_counts() / len(
    trainperf_proc["good_bad_flag"]
)

# %%
trainperf_proc["good_bad_flag"] = trainperf_proc["good_bad_flag"].replace(
    {"Good": 1, "Bad": 0}
)
trainperf_proc.good_bad_flag.value_counts()

# %% [markdown]
# ## Check null values
# Drop variables with >20% missingness and impute the rest.

# %%
traindemographics_proc.isnull().sum() / traindemographics_proc.shape[0]
traindemographics_proc.isnull().sum() / traindemographics_proc.shape[0] > 0.2
demo_drop = traindemographics_proc.columns[
    traindemographics_proc.isnull().sum() / traindemographics_proc.shape[0]
    > 0.2
]
traindemographics_proc = traindemographics_proc.drop(columns=demo_drop)
traindemographics_proc.dtypes

# %%
imp = SimpleImputer(
    strategy="most_frequent"
)  # TODO: replace with Random Forest

# %%
traindemographics_proc = pd.DataFrame(
    imp.fit_transform(X=traindemographics_proc),
    columns=traindemographics_proc.columns,
)
traindemographics_proc.dtypes

# %%
def simple_imputer(df, th=0.2):
    """
    Imputes missing variables that are less than 20% missing with the most frequent value.
    """
    demo_drop = df.columns[df.isnull().sum() / df.shape[0] > th]
    df_toimp = df.drop(columns=demo_drop)

    imp = SimpleImputer(strategy="most_frequent")

    df_imp = pd.DataFrame(
        imp.fit_transform(X=df_toimp), columns=df_toimp.columns
    )

    return df_imp


# %%
trainprevloans_proc = simple_imputer(df=trainprevloans_proc)
trainperf_proc = simple_imputer(df=trainperf_proc)

# %%
traindemographics_proc.shape
trainprevloans_proc.shape
trainperf_proc.shape

# %% [markdown]
# ## Numeric variables
# Check mean, median, and range of values.
# ### Revert data types

# %%
# "most_frequent" imputation changes numeric variables to object

num_cols = traindemographics_raw.select_dtypes(include=np.number).columns
num_repl = {k: v for k, v in zip(num_cols, ["float"] * len(num_cols))}
traindemographics_proc = traindemographics_proc.astype(dtype=num_repl)
traindemographics_proc.dtypes

# %%
def revert_numeric(proc, raw):
    """
    Reverts object data types to numeric.
    """
    num_cols = raw.select_dtypes(include=np.number).columns
    num_repl = {k: v for k, v in zip(num_cols, ["float"] * len(num_cols))}
    proc = proc.astype(dtype=num_repl)
    return proc


# %%
trainprevloans_proc = revert_numeric(
    proc=trainprevloans_proc, raw=trainprevloans_raw
)
trainperf_proc = revert_numeric(proc=trainperf_proc, raw=trainperf_raw)

# %% [markdown]
# ### Summary stats

# %%
traindemographics_proc.select_dtypes(include=np.number).head()

# %% [markdown]
# Scaling will be applied due to the wide range i.e. max value is more than 1 sd from the mean. This will be applied on the amounts.

# %%
trainprevloans_proc.select_dtypes(include=np.number).describe().transpose()

# %%
scaler_pl = StandardScaler()

sc_cols = ["loanamount", "totaldue"]
pd.DataFrame(
    scaler_pl.fit_transform(X=trainprevloans_proc[sc_cols]), columns=sc_cols
)

trainprevloans_proc = pd.concat(
    objs=[
        trainprevloans_proc.drop(columns=sc_cols),
        pd.DataFrame(
            scaler_pl.fit_transform(X=trainprevloans_proc[sc_cols]),
            columns=sc_cols,
        ),
    ],
    axis=1,
)

trainprevloans_proc

# %%
trainperf_proc.select_dtypes(include=np.number).describe().transpose()

# %%
scaler_cl = StandardScaler()

sc_cols = ["loanamount", "totaldue"]
pd.DataFrame(
    scaler_cl.fit_transform(X=trainperf_proc[sc_cols]), columns=sc_cols
)

trainperf_proc = pd.concat(
    objs=[
        trainperf_proc.drop(columns=sc_cols),
        pd.DataFrame(
            scaler_cl.fit_transform(X=trainperf_proc[sc_cols]), columns=sc_cols
        ),
    ],
    axis=1,
)

trainperf_proc

# %% [markdown]
# ## Categorical variables
# Check number of distinct values.

# %%
traindemographics_proc.select_dtypes(include=pd.Categorical).nunique()

# %% [markdown]
# * [Tier 1 banks](https://thenationonlineng.net/tier-1-banks-assets-hit-n46tr/#:~:text=The%20banks%20include%3A%20United%20Bank,of%20Nigeria%20Holdings%20(FBNH).)
# * [Tier 2 banks](https://businessday.ng/banking/article/tier-2-banks-maintains-npl-ratios-lower-than-5-0-threshold/#:~:text=Banks%20that%20fall%20under%20the,Plc%20(3.5%25)%20NPL%20ratios.)

# %%
traindemographics_proc.bank_name_clients.value_counts()  # TODO: replace with bank Tiers

# traindemographics_proc.bank_name_clients.value_counts().to_csv("../outputs/banks.csv")

# %%
trainprevloans_proc.select_dtypes(include=pd.Categorical).nunique()

# %%
trainperf_proc.select_dtypes(include=pd.Categorical).nunique()

# %% [markdown]
# ## Dates
# Change date columns to numeric.

# %%
traindemographics_proc.dtypes

# %%
customer_age = traindemographics_proc["birthdate"].apply(
    lambda x: datetime.now() - x
)
customer_age = customer_age.apply(lambda x: x.days)
traindemographics_proc["customer_age"] = customer_age

# %%
trainprevloans_proc.dtypes

# %%
trainprevloans_proc[
    "application_month"
] = trainprevloans_proc.creationdate.dt.month
trainprevloans_proc[
    "application_day"
] = trainprevloans_proc.creationdate.dt.day

# %%
trainprevloans_proc = trainprevloans_proc.drop(
    columns=trainprevloans_proc.columns[
        trainprevloans_proc.columns.str.contains("date")
    ]
)
trainprevloans_proc.head()

# %%
trainperf_proc.dtypes
trainperf_proc["application_month"] = trainperf_proc.creationdate.dt.month
trainperf_proc["application_day"] = trainperf_proc.creationdate.dt.day
trainperf_proc = trainperf_proc.drop(
    columns=trainperf_proc.columns[trainperf_proc.columns.str.contains("date")]
)
trainperf_proc.head()

# %% [markdown]
# ## Coordinates

# %%
gis = GIS()

# %%
test_geo = reverse_geocode([2.2945, 48.8583])
test_geo.get("address").get("Region")

# %%
# TODO: use administrative regions

# test_geo = traindemographics_proc.head(3)
# loc_res = {}

# for cust, long, lat in zip(test_geo.customerid, test_geo.longitude_gps, test_geo.latitude_gps):
#     try:
#         loc = reverse_geocode([long, lat])
#         loc_res[cust]=loc
#     except Exception as e:
#         loc_res[cust]=e

# loc_res

# %%
((traindemographics_proc.shape[0] / 3) * 0.9) / 60

# %% [markdown]
# # Merge datasets
# ## Summarise previous loans dataset

# %%
trainprevloans_proc.customerid.sample(n=1)
trainprevloans_proc[
    trainprevloans_proc.customerid == "8a2a81a74ce8c05d014cfb32a0da1049"
].sort_values(by="approveddate", ascending=True)

# %%
trainprevloans_summ = trainprevloans_proc.groupby(
    by="customerid", as_index=False
).aggregate(
    loans_taken=("loannumber", "max"),
    timedelta=("firstrepaiddate", lambda x: datetime.now() - x),
    avg_loanamount=("loanamount", "mean"),
)

trainprevloans_summ

# %%
trainprevloans_proc[
    trainprevloans_proc.customerid == "8a1a1e7e4f707f8b014f797718316cad"
].sort_values(by="approveddate", ascending=True)

# %%
borrow_days = trainprevloans_summ.timedelta.apply(lambda x: x.days).apply(
    lambda x: np.max(x)
)
trainprevloans_summ["borrow_days"] = borrow_days
trainprevloans_summ = trainprevloans_summ.drop(columns="timedelta")
trainprevloans_summ.head()

# %% [markdown]
# ## Merge all datasets

# %%
merge_1 = pd.merge(
    left=trainperf_proc,
    right=traindemographics_proc,
    how="left",
    on="customerid",
)
train_merged = pd.merge(
    left=merge_1, right=trainprevloans_summ, how="left", on="customerid"
)
train_merged.shape

# %%
train_merged.isna().sum()

# %%
train_merged[train_merged.birthdate.isna()]
traindemographics_proc[
    traindemographics_proc.birthdate == "8a8589f35451855401546b0738c42524"
]

# %% [markdown]
# ## Impute missing values

# %%
train_merged["loans_taken"] = train_merged.loans_taken.fillna(0)
train_merged["avg_loanamount"] = train_merged.avg_loanamount.fillna(0)
train_merged["borrow_days"] = train_merged.borrow_days.fillna(0)
train_merged["customer_age"] = train_merged.customer_age.fillna(
    np.median(traindemographics_proc.customer_age)
)
