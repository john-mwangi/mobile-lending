import os
import dill

import numpy as np
import pandas as pd

from typing import Tuple
from datetime import datetime
from dataclasses import dataclass

from utils import Scorer

CURRENT_LOANS = "../inputs/train/trainperf.csv"
DEMOGRAPHIC_DATA = "../inputs/train/traindemographics.csv"
PREVIOUS_LOANS = "../inputs/train/trainprevloans.csv"
PICKLE_DIR = "../outputs/"


@dataclass
class DataPrep:
    """This class handles data loading and preparation."""

    def __init__(
        self,
        trainperf_proc=CURRENT_LOANS,
        traindemographics_proc=DEMOGRAPHIC_DATA,
        trainprevloans=PREVIOUS_LOANS,
    ) -> None:

        self.trainperf_proc = trainperf_proc
        self.traindemographics_proc = traindemographics_proc
        self.trainprevloans = trainprevloans

    def load_customer_datasets(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads and returns the following datasets:
        current customer loans,
        demographic information about a customer,
        customers loan history.
        """

        trainperf_proc = pd.read_csv(self.trainperf_proc)
        traindemographics_proc = pd.read_csv(self.traindemographics_proc)
        trainprevloans = pd.read_csv(self.trainprevloans)

        return trainperf_proc, traindemographics_proc, trainprevloans

    @staticmethod
    def load_data_objs(items: list) -> list:
        """
        items: a list of items to retrieve
        """
        with open(
            file=os.path.join(PICKLE_DIR, "api_data.pkl"), mode="rb"
        ) as f:
            data_objs = dill.load(file=f)

            res = [
                data_objs.get(item, f"Item '{item}' not found")
                for item in items
            ]

        return res

    @staticmethod
    def convert_dates(data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts date columns to datetime.
        :param data: dataframe of raw data
        """
        date_cols = data.columns[data.columns.str.contains("date")]
        date_data = data[date_cols].transform(func=pd.to_datetime)
        trim_data = data.drop(columns=date_cols)
        new_data = pd.concat(objs=[trim_data, date_data], axis=1)
        return new_data

    @staticmethod
    def summ_numeric(data: pd.DataFrame) -> pd.DataFrame:
        """
        Summarises key numeric information.
        :param data: The output of convert_dates function.
        """
        data_summ = data.groupby(by="customerid", as_index=False).aggregate(
            loans_taken=("loannumber", "max"),
            avg_loanamount=("loanamount", "mean"),
            avg_term=("termdays", "mean"),
        )

        return data_summ

    @staticmethod
    def summ_dates(data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts certain dates to their numeric equivalents and calculates daily
        interest rates.
        :param data: The output of summ_numeric function.
        """
        data["creation_month"] = data.creationdate.dt.month
        data["creation_day"] = data.creationdate.dt.day

        firstrepaid_days = data.firstrepaiddate - data.approveddate
        firstrepaid_days = firstrepaid_days.apply(lambda x: x.days)
        data["firstrepaid_days"] = firstrepaid_days

        interest_days = (data.closeddate - data.approveddate).apply(
            lambda x: x.days
        )
        interest_amt = data.totaldue - data.loanamount
        daily_int = interest_amt / interest_days
        daily_int = daily_int.replace({float("inf"): 0})
        data["daily_int"] = daily_int

        return data

    @staticmethod
    def group_days(
        prev_loans: pd.DataFrame, loans_summ: pd.DataFrame
    ) -> pd.DataFrame:
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

    @staticmethod
    def merge_datasets(perf_df, demo_df, summ_df):
        """
        Merges customers demographic data and loan history.
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

    @staticmethod
    def revert_numeric(proc: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        """
        'most_frequent' imputation changes numeric variables to object.
        This fixes the issue.
        """
        num_cols = np.intersect1d(
            ar1=raw.select_dtypes(include=np.number).columns, ar2=proc.columns
        )

        num_repl = {k: v for k, v in zip(num_cols, ["float"] * len(num_cols))}
        proc = proc.astype(dtype=num_repl)
        return proc

    @staticmethod
    def scale_data(data: pd.DataFrame, scaler, sc_cols: list) -> pd.DataFrame:
        """
        Scales test data.
        :param testdata: dataframe containing test data
        :param scaler: a fitted scaler
        :param sc_cols: a list of columns to scale
        """

        data_sc = pd.concat(
            objs=[
                data.drop(columns=sc_cols),
                pd.DataFrame(
                    scaler.fit_transform(X=data[sc_cols]), columns=sc_cols
                ),
            ],
            axis=1,
        )

        return data_sc

    @staticmethod
    def customer_age(df: pd.DataFrame) -> pd.DataFrame:
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


if __name__ == "__main__":

    dp = DataPrep()
    sc = Scorer()

    (
        trainperf_proc,
        traindemographics_proc,
        trainprevloans_raw,
    ) = dp.load_customer_datasets()

    print("[*] Data importation")
    assert trainperf_proc.shape == (4368, 10)
    assert traindemographics_proc.shape == (4346, 9)
    assert trainprevloans_raw.shape == (18183, 12)

    print("\n[*] Convert date cols to date")
    trainprevloans_proc = dp.convert_dates(trainprevloans_raw)
    print(trainprevloans_proc.dtypes)

    print("\n[*] Summarise numeric cols of prev loans")
    trainprevloans_summ = dp.summ_numeric(data=trainprevloans_proc)
    assert trainprevloans_summ.shape == (4359, 4)

    print("\n[*] Calculate daily int rate")
    trainprevloans_proc = dp.summ_dates(data=trainprevloans_proc)
    assert trainprevloans_proc.shape == (18183, 16)

    print("\n[*] Summarise date cols")
    trainprevloans_summ = dp.group_days(
        prev_loans=trainprevloans_proc, loans_summ=trainprevloans_summ
    )
    assert trainprevloans_summ.shape == (4359, 8)

    print("\n[*] Merge all customer's information")
    train_merged_proc = dp.merge_datasets(
        perf_df=trainperf_proc,
        demo_df=traindemographics_proc,
        summ_df=trainprevloans_summ,
    )

    train_merged_raw = dp.convert_dates(data=train_merged_proc)
    train_merged_proc = train_merged_raw.copy()

    assert train_merged_proc.shape == (4368, 25)

    print("\n[*] Drop cols that were identified as null")
    null_cols = dp.load_data_objs(items=["null_cols"])
    null_cols = null_cols[0]

    train_merged_proc = train_merged_proc.drop(columns=null_cols)

    assert train_merged_proc.shape == (4368, 21)

    print("\n[*] Impute missing values")
    imputer = dp.load_data_objs(items=["imputer"])
    imputer = imputer[0]

    try:
        train_imp = train_merged_proc.drop(columns="good_bad_flag")
    except:
        train_imp = train_merged_proc

    train_merged_proc = pd.DataFrame(
        imputer.transform(X=train_imp), columns=train_imp.columns
    )

    assert train_merged_proc.shape == (4368, 20)

    print("\n[*] Revert numeric data types")
    train_merged_proc = dp.revert_numeric(
        proc=train_merged_proc, raw=train_merged_raw
    )

    print(train_merged_proc.dtypes)

    print("\n[*] Scale the data")
    scaler, sc_cols = dp.load_data_objs(items=["scaler", "scale_cols"])

    train_merged_proc = dp.scale_data(
        data=train_merged_proc, scaler=scaler, sc_cols=sc_cols
    )

    assert train_merged_proc.shape == (4368, 20)

    print("\n[*] Calculate customer age")
    train_merged_proc = dp.customer_age(df=train_merged_proc)
    assert train_merged_proc.shape == (4368, 18)

    print("\n[*] Drop id vars")
    id_vars = train_merged_proc.columns[
        train_merged_proc.columns.str.endswith("id")
    ]
    train_merged_proc = train_merged_proc.drop(columns=id_vars)
    assert train_merged_proc.shape == (4368, 16)

    print("\n[*] Dummification")
    dummer_df = dp.load_data_objs(items=["dummer_df"])
    dummer_df = dummer_df[0]

    train_dumm_df = pd.concat(objs=[train_merged_proc, dummer_df], axis=0)
    train_merged_proc_X = pd.get_dummies(train_dumm_df, drop_first=False)
    train_merged_proc_X = train_merged_proc_X.iloc[
        : train_merged_proc.shape[0], :
    ]

    assert train_merged_proc_X.shape == (4368, 35)

    print("\n[*] Dataset has been successfully processed")
    print(f"Shape: {train_merged_proc_X.shape}")

    train_X_json = train_merged_proc_X.sample(n=5).to_json()
    print(type(train_X_json))
    print(type(train_merged_proc_X))
