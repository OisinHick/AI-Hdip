Pandas basics

This is a short introduction to pandas, geared mainly for new users. You can see more complex recipes in the Cookbook.

This tutorial is based on the official guide

To download this notebook click on this link

First we need to import the following packages:

import numpy as np
import pandas as pd

1. Object Creation

Creating a Series by passing a list of values, letting pandas create a default RangeIndex.

s = pd.Series([1, 3, 5, np.nan, 6, 8])
s

Creating a DataFrame by passing a NumPy array, with a datetime index using date_range() and labeled columns:

dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
df

Creating a DataFrame by passing a dictionary of objects that can be converted into a series-like structure:

df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)
df2

The columns of the resulting DataFrame have different dtypes:

df2.dtypes

2. Viewing Data

Use DataFrame.head() and DataFrame.tail() to view the top and bottom rows of the frame respectively:

df.head()

df.tail(3)

Display the DataFrame.index or DataFrame.columns:

df.index

df.columns

Return a NumPy representation of the underlying data with DataFrame.to_numpy() without the index or column labels:

df.to_numpy()

describe() shows a quick statistic summary of your data:

df.describe()

Transposing your data:

df.T

Sorting by an axis via DataFrame.sort_index():

df.sort_index(axis=1, ascending=False)

Sorting by values via DataFrame.sort_values():

df.sort_values(by="B")

3. Selection

Note: For production code, optimized pandas data access methods (.loc, .iloc, .at, .iat) are recommended.
Getting ([])

Selecting a single column, which yields a Series:

df["A"]

Selecting via [] ([row:row]), which slices the rows:

df[0:3]

df["20130102":"20130104"]

Selection by Label (.loc / .at)

For getting a cross section using a label:

df.loc[dates[0]]

Selecting on a multi-axis by label:

df.loc[:, ["A", "B"]]

Showing label slicing, both endpoints are included:

df.loc["20130102":"20130104", ["A", "B"]]

For getting fast access to a scalar:

df.at[dates[0], "A"]

Selection by Position (.iloc / .iat)

Select via the position of the passed integers:

df.iloc[3]

By integer slices, acting similar to numpy/python:

df.iloc[3:5, 0:2]

By lists of integer position locations, similar to numpy/python:

df.iloc[[1, 2, 4], [0, 2]]

For getting fast access to a scalar positionally:

df.iat[1, 1]

Boolean Indexing

Using a single column’s values to select data:

df[df["A"] > 0]

Selecting values from a DataFrame where a boolean condition is met:

df[df > 0]

Using isin() method for filtering:

df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
df2[df2["E"].isin(["two", "four"])]

4. Missing Data

pandas primarily uses the value np.nan to represent missing data. It is by default not included in computations.

Reindexing allows you to change/add/delete the index on a specified axis:

df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1.loc[dates[0] : dates[1], "E"] = 1
df1

To drop any rows that have missing data:

df1.dropna(how="any")

Filling missing data:

df1.fillna(value=5)

To get the boolean mask where values are NaN:

pd.isna(df1)

5. Operations
Stats

Operations in general exclude missing data.

Performing a descriptive statistic along columns:

df.mean()

Same operation on the other axis (rows):

df.mean(axis=1)

User-Defined Functions (UDFs)

Applying functions to the data using apply():

df.apply(lambda x: x.max() - x.min())

6. Merge & Grouping
Concat

Concatenating pandas objects together with concat():

df_concat = pd.DataFrame(np.random.randn(10, 4))
pieces = [df_concat[:3], df_concat[3:7], df_concat[7:]]
pd.concat(pieces)

Join

SQL-style merges using merge():

left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
pd.merge(left, right, on="key")

Grouping

By “group by” we are referring to a process involving one or more of the following steps:

    Splitting the data into groups based on some criteria
    Applying a function to each group independently
    Combining the results into a data structure

df_group = pd.DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
        "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
        "C": np.random.randn(8),
        "D": np.random.randn(8),
    }
)
df_group.groupby("A")[["C", "D"]].sum()

7. Reshaping & Pivot Tables

Pivot tables can be created easily using pivot_table():

df_pivot = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 3,
        "B": ["A", "B", "C"] * 4,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
        "D": np.random.randn(12),
        "E": np.random.randn(12),
    }
)

pd.pivot_table(df_pivot, values="D", index=["A", "B"], columns=["C"])

8. Time Series & Categoricals
Time Series

Resampling time series data:

rng = pd.date_range("1/1/2012", periods=100, freq="s")
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample("5Min").sum()

Categoricals

pandas can include categorical data in a DataFrame:

df_cat = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
)
df_cat["grade"] = df_cat["raw_grade"].astype("category")
df_cat["grade"]

9. Plotting

Standard plotting with matplotlib integration:

import matplotlib.pyplot as plt
plt.close("all")

ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
ts = ts.cumsum()
ts.plot()

10. Getting Data In/Out
CSV

Writing to a CSV file:

df.to_csv("foo.csv")

Reading from a CSV file:

pd.read_csv("foo.csv")