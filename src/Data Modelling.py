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
    Converts date columns to datetime
    """
    date_cols = df.columns[df.columns.str.contains("date")]
    date_df = df[date_cols].transform(func=pd.to_datetime)
    trim_df = df.drop(columns=date_cols)
    new_df = pd.concat(objs=[trim_df, date_df], axis=1)
    return new_df


# %%
trainprevloans_proc = convert_dates(trainprevloans_proc)
trainprevloans_proc.dtypes

# %% [markdown]
# ### Summarise numeric columns

# %%
trainprevloans_summ = trainprevloans_proc.groupby(
    by="customerid", as_index=False
).aggregate(
    loans_taken=("loannumber", "max"),
    avg_loanamount=("loanamount", "mean"),
    avg_term=("termdays", "mean"),
)

trainprevloans_summ.head()

# %% [markdown]
# ### Summarise date columns

# %%
trainprevloans_proc.head()

# %% [markdown]
# Date transformations:
# * approveddate: none, creationdate will be used
# * creationdate: does the month of year and day of month affect repayment?
# * closeddate: none, reflected in default
# * firstduedate: none
# * firstrepaiddate: how soon does the customer pay after loan approval (sign of stress?)

# %%
trainprevloans_proc[
    "creation_month"
] = trainprevloans_proc.creationdate.dt.month
trainprevloans_proc["creation_day"] = trainprevloans_proc.creationdate.dt.day

# %%
firstrepaid_days = (
    trainprevloans_proc.firstrepaiddate - trainprevloans_proc.approveddate
)
firstrepaid_days = firstrepaid_days.apply(lambda x: x.days)
trainprevloans_proc["firstrepaid_days"] = firstrepaid_days

# %%
# Historic daily customer interest rates
interest_days = (
    trainprevloans_proc.closeddate - trainprevloans_proc.approveddate
).apply(lambda x: x.days)
interest_amt = trainprevloans_proc.totaldue - trainprevloans_proc.loanamount
daily_int = interest_amt / interest_days
daily_int = daily_int.replace({float("inf"): 0})
trainprevloans_proc["daily_int"] = daily_int

# %%
trainprevloans_summ = (
    trainprevloans_proc.groupby(by="customerid", as_index=False)
    .aggregate(
        avg_creation_month=("creation_month", "mean"),
        avg_creation_day=("creation_day", "mean"),
        avg_firstrepaid_days=("firstrepaid_days", "mean"),
        avg_daily_int=("daily_int", "mean"),
    )
    .merge(right=trainprevloans_summ, how="left", on="customerid")
)

trainprevloans_summ.shape

# %%
trainprevloans_summ.head()

# %% [markdown]
# ## Merge all datasets
# ### Check duplicates

# %%
len(trainperf_proc.customerid) == len(pd.unique(trainperf_proc.customerid))
len(traindemographics_proc.customerid) == len(
    pd.unique(traindemographics_proc.customerid)
)

# %%
traindemographics_proc = traindemographics_proc.drop_duplicates(
    subset=["customerid"]
)

# %% [markdown]
# ### Merge datasets

# %%
merge_1 = pd.merge(
    left=trainperf_proc,
    right=traindemographics_proc,
    how="left",
    on="customerid",
)
train_merged_raw = pd.merge(
    left=merge_1, right=trainprevloans_summ, how="left", on="customerid"
)
train_merged_raw.shape

train_merged_proc = train_merged_raw.copy()
train_merged_proc.shape

# %% [markdown]
# # EDA
# ## Fix data types

# %%
train_merged_proc.head()

# %%
train_merged_proc.dtypes

# %%
train_merged_proc = convert_dates(train_merged_proc)

# %% [markdown]
# ## Outcome variable

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
# ## Check null values
# Drop variables with >20% missingness and impute the rest.

# %%
train_merged_proc.isnull().sum() / train_merged_proc.shape[0]

# %%
train_merged_proc.isnull().sum() / train_merged_proc.shape[0] > 0.3
cols_drop = train_merged_proc.columns[
    train_merged_proc.isnull().sum() / train_merged_proc.shape[0] > 0.3
]
train_merged_proc = train_merged_proc.drop(columns=cols_drop)
train_merged_proc.shape

# %%
imp = SimpleImputer(
    strategy="most_frequent"
)  # TODO: replace with Random Forest

# %%
train_merged_proc = pd.DataFrame(
    imp.fit_transform(X=train_merged_proc), columns=train_merged_proc.columns
)
train_merged_proc.shape

# %% [markdown]
# ## Numeric variables
# Check mean, median, and range of values.
# ### Revert data types

# %%
train_merged_proc.dtypes

# %%
# "most_frequent" imputation changes numeric variables to object
num_cols = np.intersect1d(
    ar1=train_merged_raw.select_dtypes(include=np.number).columns,
    ar2=train_merged_proc.columns,
)

num_repl = {k: v for k, v in zip(num_cols, ["float"] * len(num_cols))}
train_merged_proc = train_merged_proc.astype(dtype=num_repl)
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
pd.DataFrame(
    scaler.fit_transform(X=train_merged_proc[sc_cols]), columns=sc_cols
)

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
# ## Dates
# Change date columns to numeric.

# %%
train_merged_proc.select_dtypes(include="datetime64").columns

# %%
customer_age = train_merged_proc.birthdate.apply(
    lambda x: datetime.now() - x
).apply(lambda x: x.days)

train_merged_proc["customer_age"] = customer_age

# %%
train_merged_proc = train_merged_proc.drop(
    columns=train_merged_proc.select_dtypes(include="datetime64").columns
)

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
