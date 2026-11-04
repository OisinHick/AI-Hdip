Lab 4 Lab Statistical Reasoning and Pandas for EDA

Module: Data Handling and Infrastructure for AI
Platform: Google Colab
Dataset: Titanic passenger data
Duration: approximately 2 hours

The objective is to create a reproducible chain from an analytical question to a calculation, visualisation, interpretation and pipeline requirement.
Learning outcomes

You will inspect semantic variable roles, calculate resistant summaries, diagnose missingness, compare groups using effect sizes, construct a bootstrap interval, examine conditional patterns and document dependence, distribution-shift and leakage risks.
Schedule
Activity 	Time
Setup and loading 	10 min
Schema and semantic roles 	15 min
Summaries and distributions 	25 min
Missingness 	15 min
Group comparison and association 	20 min
Bootstrap uncertainty 	15 min
Conditional analysis and risks 	10 min
Independent challenge 	10 min
1 Colab setup

Use Runtime then Run all after completing the TODO cells. Colab already contains Python; the following command ensures that the scientific packages are available.

%pip install -q pandas numpy scipy matplotlib seaborn statsmodels

import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

SEED = 42
rng = np.random.default_rng(SEED)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda value: f"{value:,.3f}")
sns.set_theme(style="whitegrid", context="notebook")

print("Python:", platform.python_version())
print("pandas:", pd.__version__)
print("NumPy:", np.__version__)

A fixed seed makes resampling repeatable. Recording package versions helps another analyst reproduce the computational environment.
2 Load and validate the data

The public URL is a common fallback. Students who loaded the file into GCS during Sprint 1 may replace the URL-loading cell with their authenticated GCS code while retaining the dataframe name df.

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)

print(f"Rows: {df.shape[0]:,}; columns: {df.shape[1]}")
display(df.head())

required = {"PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
            "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"}
assert required.issubset(df.columns)
assert df["PassengerId"].is_unique
assert set(df["Survived"].unique()).issubset({0, 1})
assert set(df["Pclass"].unique()).issubset({1, 2, 3})
print("Initial data-contract checks passed.")

3 Schema and semantic-role audit

Pandas dtypes describe storage. Analytical roles describe meaning. Pclass is stored as an integer but is ordinal categorical; PassengerId is stored as an integer but is an identifier.

schema_audit = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null": df.notna().sum(),
    "missing": df.isna().sum(),
    "unique": df.nunique(dropna=True)
}).sort_values("missing", ascending=False)
display(schema_audit)

Exercise 1 (Semantic roles)

Create a table for PassengerId, Survived, Pclass, Age, Fare, Sex, Embarked and Ticket. For each variable, state its analytical role and a suitable summary. Explain why the mean of PassengerId is meaningless.

variable_roles = pd.DataFrame({
    "variable": [],
    "analytical_role": [],
    "suitable_summary": []
})
display(variable_roles)

4 Numerical summaries

Mean and standard deviation are sensitive to extremes. Median and IQR are more resistant. Calculate both pairs before deciding which best communicates the distribution.

def profile(series):
    observed = series.dropna()
    q1, q3 = observed.quantile([0.25, 0.75])
    return pd.Series({
        "n": observed.size,
        "missing": series.isna().sum(),
        "mean": observed.mean(),
        "median": observed.median(),
        "std": observed.std(),
        "IQR": q3 - q1,
        "min": observed.min(),
        "max": observed.max(),
        "skew": observed.skew()
    })

summary = df[["Age", "Fare", "SibSp", "Parch"]].apply(profile).T
display(summary)

Exercise 2 (Interpretation)

Write two evidence-based findings. Identify the most strongly right-skewed variable and explain whether mean and standard deviation or median and IQR are more informative.

finding_1 = ""
finding_2 = ""
print("Finding 1:", finding_1)
print("Finding 2:", finding_2)

5 Distribution plots and transformation

A histogram exposes shape; a boxplot shows quartiles and flagged extremes. A flagged value is not automatically erroneous.

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
sns.histplot(data=df, x="Age", bins=20, kde=True, ax=axes[0, 0], color="#54B7AF")
sns.histplot(data=df, x="Fare", bins=30, kde=True, ax=axes[0, 1], color="#FF9D38")
sns.boxplot(data=df, x="Age", ax=axes[1, 0], color="#54B7AF")
sns.boxplot(data=df, x="Fare", ax=axes[1, 1], color="#FF9D38")
axes[0, 0].set_title("Observed age distribution")
axes[0, 1].set_title("Observed fare distribution")
axes[1, 0].set_title("Age boxplot")
axes[1, 1].set_title("Fare boxplot")
plt.tight_layout()
plt.show()

Exercise 3 (Log transformation)

Create LogFare using log(1 + Fare). Plot Fare and LogFare side by side. Explain what becomes easier to see and what becomes less directly interpretable.

df = df.assign(LogFare=...)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Plot Fare on axes[0] and LogFare on axes[1].
plt.tight_layout()
plt.show()

6 Categorical summaries and majority baseline

Counts show evidential support. Proportions enable comparison. The largest target-class proportion is a basic accuracy baseline.

target = df["Survived"].value_counts(dropna=False).to_frame("count")
target["proportion"] = target["count"] / len(df)
display(target)
print(f"Majority-class accuracy baseline: {target['proportion'].max():.1%}")

Exercise 4 (Categorical profiles)

Produce count and proportion tables for Pclass, Sex and Embarked. Retain missing values and identify the denominator.

for column in ["Pclass", "Sex", "Embarked"]:
    # Build and display count and proportion columns.
    pass

7 Missingness extent and structure

A missing percentage measures extent. Grouped missingness rates provide evidence about structure, but cannot prove MCAR, MAR or MNAR.

missing = df.isna().mean().mul(100).sort_values(ascending=False).rename("missing_percent")
display(missing.to_frame())

ax = missing[missing > 0].sort_values().plot(kind="barh", color="#E64141", figsize=(8, 4))
ax.set_xlabel("Missing values (%)")
ax.set_title("Missingness extent")
plt.tight_layout()
plt.show()

Exercise 5 (Age missingness by passenger class)

Create an AgeMissing indicator. Calculate passenger count and missingness rate within each class. Explain what a difference indicates and what it does not prove.

df = df.assign(AgeMissing=...)
age_missing_by_class = (
    df.groupby("Pclass")
      .agg(passengers=("PassengerId", "size"),
           age_missing_rate=("AgeMissing", "mean"))
)
display(age_missing_by_class)

8 Group comparison and effect size

Report counts and within-group rates before calculating the percentage-point risk difference. Treat the result as an observed association, not an isolated causal effect.

survival_by_sex = (
    df.groupby("Sex", observed=True)
      .agg(passengers=("PassengerId", "size"),
           survivors=("Survived", "sum"),
           survival_rate=("Survived", "mean"))
)
survival_by_sex["survival_percent"] = 100 * survival_by_sex["survival_rate"]
display(survival_by_sex)

difference = survival_by_sex.loc["female", "survival_rate"] - survival_by_sex.loc["male", "survival_rate"]
print(f"Observed risk difference: {100*difference:.1f} percentage points")

Exercise 6 (Survival by class)

Calculate count and survival rate by Pclass. Create a labelled bar chart with a 0–100 percent y-axis. Write one conclusion and one limitation.

survival_by_class = ...
display(survival_by_class)
# Add the bar chart.

9 Chi-square and Cramer’s V

The contingency table is the evidence. Chi-square evaluates compatibility with independence; Cramer’s V summarises association magnitude. Neither establishes causality.

table = pd.crosstab(df["Pclass"], df["Survived"])
display(table)

chi2, p_value, dof, expected = stats.chi2_contingency(table)
n = table.to_numpy().sum()
rows, columns = table.shape
cramers_v = np.sqrt((chi2 / n) / min(columns - 1, rows - 1))

print(f"Chi-square: {chi2:.3f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.3e}")
print(f"Cramer's V: {cramers_v:.3f}")
display(pd.DataFrame(expected, index=table.index, columns=table.columns))

Interpret the expected counts, p-value and Cramer’s V. Explain why sample size affects the p-value and why the result is not causal.
10 Bootstrap uncertainty

Bootstrap the female-minus-male survival-rate difference. If observations were clustered by household, the household rather than the row might need to be resampled.

def survival_difference(data):
    # TODO 7A
    pass

observed = survival_difference(df)
boot = []
for _ in range(2000):
    # TODO 7B: sample len(df) rows with replacement.
    sample = ...
    boot.append(survival_difference(sample))

boot = np.asarray(boot)
# TODO 7C: calculate the 2.5% and 97.5% quantiles.
lower, upper = ...

print(f"Observed difference: {observed:.3f}")
print(f"95% percentile bootstrap interval: [{lower:.3f}, {upper:.3f}]")
# TODO 7D: plot the bootstrap distribution and interval limits.

11 Conditional analysis

Compare marginal survival rates with class-conditional rates. Retain counts so that unstable small groups remain visible.

conditional = (
    df.groupby(["Pclass", "Sex"], observed=True)
      .agg(passengers=("PassengerId", "size"),
           survival_rate=("Survived", "mean"))
      .assign(survival_percent=lambda x: 100 * x["survival_rate"])
      .reset_index()
)
display(conditional)

conditional.pivot(index="Pclass", columns="Sex", values="survival_percent").plot(
    kind="bar", figsize=(8, 4), color=["#54B7AF", "#5B63F0"]
)
plt.ylabel("Observed survival rate (%)")
plt.xlabel("Passenger class")
plt.title("Observed survival by passenger class and sex")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

Exercise 8 (Interpretation)

Does the recorded-sex difference remain within each class? Are group sizes similar? How does conditioning change the interpretation? Why is the result still not causal?

answer_1 = ""
answer_2 = ""
answer_3 = ""
answer_4 = ""

12 Dependence shift and leakage

Create a risk table for your project. Consider related entities, time order, source changes, feature-availability time and preprocessing fitted before splitting.

project_risks = pd.DataFrame({
    "risk": [],
    "evidence_to_check": [],
    "planned_response": []
})
display(project_risks)

13 Independent challenge (this is optional, just for individual learning)

Choose one:

    Age and survival: compare distributions using count, median, IQR and a suitable plot. Discuss missing ages.
    Family size: create FamilySize = SibSp + Parch + 1 and compare survival across justified groups.
    Embarkation: compare survival by Embarked, retaining missing values and reporting counts.

Provide the question and estimand, analytical population, table, visualisation, two findings, two limitations and one pipeline requirement.

# TODO 9: Complete one independent challenge.
# Add code and Markdown cells as needed.