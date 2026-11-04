Lab 5: Exploratory Data Analysis with pandas, NumPy and scikit-learn

Data Handling and Infrastructure for AI | MSc | 2 hours
How this notebook works

    Run the cells in order, from top to bottom.
    Most code is already written. In each task you complete a few blanks marked ___ (until you do, you will see a NameError: that is expected).
    After each task a self-check cell prints PASS or CHECK AGAIN. If you see CHECK AGAIN, re-read your blanks. Use the hint if you are stuck, or just ask.
    Some tasks end with a short written answer (one or two sentences, double-click the cell to edit). Write in your own words.

Setup

Run the two cells below, one after the other. The first imports everything and defines check. The second loads the data and keeps an untouched copy, df_raw.

# --- Setup: run once ---------------------------------------------------------
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 110)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (7, 3.4)

def check(label, ok):
    """Self-check: prints PASS or CHECK AGAIN."""
    print(("PASS         " if bool(ok) else "CHECK AGAIN  ") + label)

# --- Load the data (provided) --------------------------------------------------
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
try:
    df = pd.read_csv(URL)
except Exception:
    # If the download is blocked: upload titanic.csv with the file panel on the left, then:
    df = pd.read_csv("titanic.csv")

df_raw = df.copy()        # untouched copy
print("shape:", df.shape)
df.head(3)

Task 1. Write a data contract

A data contract is code that states what the data should look like, so that you find out immediately if it ever stops being true. For example set(df["Survived"]) <= {0, 1}.

Complete the three blanks, using the same pattern. (PassengerId is done for you.)

def contract(frame):
    """Return {rule: True/False}. All True means the file matches the contract."""
    return {
        "PassengerId is unique":   frame["PassengerId"].is_unique,
        "Survived is 0 or 1":      ___,
        "Pclass is 1, 2 or 3":     ___,
        "Sex is male or female":   ___,
    }

contract(df)

check("the real file satisfies the contract", all(contract(df).values()))

# Now we corrupt a COPY on purpose and see whether the contract notices (provided):
bad = df.copy()
bad.loc[0, "Pclass"] = 4
print(contract(bad))
check("the contract catches the corrupted copy", not all(contract(bad).values()))

Task 2. How much is missing, and for whom?

Measure before you decide. First, the percentage of missing values per column. Then, the missing rate of Age and Cabin for each passenger class (the groupby pattern from slide 17: the mean of a True/False column is a proportion).

Part A. Fill in the first blank to get the percentage missing per column (rounded to 1 decimal). Part B. Fill in the two isna() blanks.

missing_pct = ___                 # percent missing per column, rounded to 1 decimal
display(missing_pct[missing_pct > 0].sort_values(ascending=False))

by_class = (
    df.assign(age_na=___, cabin_na=___)
      .groupby("Pclass")[["age_na", "cabin_na"]]
      .mean()
      .round(3)
)
display(by_class)

by_class.mul(100).plot.bar(rot=0, color=["#FF9D38", "#E64141"])
plt.ylabel("% missing"); plt.title("Missing rate by passenger class"); plt.show()

Hint

check("Cabin 77.1% and Age 19.9% missing", missing_pct["Cabin"] == 77.1 and missing_pct["Age"] == 19.9)
check("Age missing for 27.7% of 3rd class", abs(by_class.loc[3, "age_na"] - 0.277) < 0.002)
check("Cabin missing for 18.5% of 1st class", abs(by_class.loc[1, "cabin_na"] - 0.185) < 0.002)

Your answer (1 to 2 sentences, double-click to edit). Look at the plot. Which missingness mechanism (MCAR, MAR, MNAR) does it make implausible, and what can you not conclude?


Task 3. Check validity and extreme values

Part A (just run it). A validation log: each rule, its rationale, and how many rows break it. Notice the last rule is a plausibility check.

Part B (complete the blanks). Compute the 1.5 x IQR fence for Fare, count how many passengers it flags, and the share of all first-class passengers that would be removed.

# Part A: validation log (provided)
rules = {
    "PassengerId is unique":        ("Row key",                    lambda d: int(d["PassengerId"].duplicated().sum())),
    "Age between 0 and 100":        ("Plausible human range",      lambda d: int((~d["Age"].dropna().between(0, 100)).sum())),
    "Fare is at least 0":           ("Prices cannot be negative",  lambda d: int((d["Fare"] < 0).sum())),
    "Pclass in {1, 2, 3}":          ("Data contract",              lambda d: int((~d["Pclass"].isin([1, 2, 3])).sum())),
    "Fare equals 0 (plausibility)": ("Free passage? crew? unknown", lambda d: int((d["Fare"] == 0).sum())),
}
log = pd.DataFrame([{"rule": k, "rationale": v[0], "violations": v[1](df)} for k, v in rules.items()])
log

# Part B: extreme values
q1, q3 = ___, ___                     # 25th and 75th percentiles of Fare (use .quantile)
fence = ___                           # q3 + 1.5 * (q3 - q1)
flagged = df[df["Fare"] > fence]
share_first = ___                     # flagged first-class passengers / ALL first-class passengers

print(f"fence = {fence:.1f}; flagged = {len(flagged)} ({len(flagged) / len(df):.0%} of passengers)")
print(f"share of first class flagged: {share_first:.0%}")
print("skewness: raw", round(df["Fare"].skew(), 2), "| after log1p", round(np.log1p(df["Fare"]).skew(), 2))

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].hist(df["Fare"], bins=40, color="#3431B6"); ax[0].axvline(fence, color="#E64141", ls="--"); ax[0].set_title("Fare (raw)")
ax[1].hist(np.log1p(df["Fare"]), bins=40, color="#54B7AF"); ax[1].set_title("log(1 + Fare)")
plt.tight_layout(); plt.show()

Hint

check("the validation log finds 15 zero fares and no other violations", log["violations"].tolist() == [0, 0, 0, 0, 15])
check("116 passengers flagged", len(flagged) == 116)
check("about 48% of first class flagged", abs(share_first - 0.481) < 0.01)

Your answer (1 to 2 sentences). Would you delete the flagged fares? Say why or why not (slides 21 and 22).
Task 4. The ECDF and the majority baseline

ECDF : sort the observed ages, rank them 1 to n, divide by n. The result at any age is the proportion of observed ages at or below it. Remember n is 714 (the missing ages are not in it).

Baseline: the accuracy of a model that always predicts the most common class. Any real model must beat it.

age = ___                          # the observed ages, sorted, as a NumPy array
ecdf = ___                         # 1/n, 2/n, ..., n/n
plt.step(age, ecdf, where="post", color="#3431B6")
plt.xlabel("Age (years)"); plt.ylabel("Cumulative proportion"); plt.title("ECDF of observed ages"); plt.show()

F28 = (age <= 28).mean()
print("n =", len(age), "| F(28) =", round(F28, 3))

counts = df["Survived"].value_counts()
baseline = ___                     # largest class count divided by the total
print(counts.to_dict(), "| majority baseline accuracy:", round(baseline, 3))

Hint

check("n = 714 observed ages", len(age) == 714)
check("the ECDF reaches exactly 1", ecdf[-1] == 1.0)
check("majority baseline is 0.616", abs(baseline - 549 / 891) < 1e-9)

Task 5. Compare groups: effect size and uncertainty

Part A (just run it). Age by outcome: similar medians, different shapes.

Part B: Cramer’s V. A p-value only says independence is implausible; V tells you how strong the association is (0 to 1): V = sqrt( chi2 / (n * (min(rows, cols) - 1)) ). Complete the three blanks.

Part C: confidence intervals A rate without its denominator and an interval is an anecdote. Use proportion_confint with the Wilson method.

# Part A (provided): Age by outcome. Same median, different shape.
display(df.groupby("Survived")["Age"].agg(["count", "mean", "median", "std"]).round(2))
sns.violinplot(data=df, x="Survived", y="Age", inner="quartile", cut=0,
               hue="Survived", palette=["#E64141", "#54B7AF"], legend=False)
plt.title("Age by outcome (survivors include more small children)"); plt.show()

# Part B: Cramer's V
def cramers_v(table):
    chi2, p, dof, _ = st.chi2_contingency(table)
    n = ___                              # total count in the table
    k = ___                              # min(rows, columns) - 1
    return ___                           # the formula for V

t_class = pd.crosstab(df["Pclass"], df["Survived"])
t_sex   = pd.crosstab(df["Sex"], df["Survived"])
print(f"V(Pclass, Survived) = {cramers_v(t_class):.3f}")
print(f"V(Sex, Survived)    = {cramers_v(t_sex):.3f}")

# Part C: Wilson 95% intervals for survival by class
g = df.groupby("Pclass")["Survived"].agg(k="sum", n="size")
g["rate"] = ___                          # k / n
g[["lo", "hi"]] = [proportion_confint(k, n, method=___) for k, n in zip(g["k"], g["n"])]
g.round(3)

Hint

check("V(Pclass) = 0.340", abs(cramers_v(t_class) - 0.340) < 0.002)
check("V(Sex) = 0.541", abs(cramers_v(t_sex) - 0.541) < 0.002)
check("3rd class: 24.2% survived, interval 20.7% to 28.2%", abs(g.loc[3, "rate"] - 0.242) < 0.002 and abs(g.loc[3, "lo"] - 0.207) < 0.002 and abs(g.loc[3, "hi"] - 0.282) < 0.002)

Your answer (1 to 2 sentences). Both associations have tiny p-values. Which is stronger, and what does that tell you about what a p-value can and cannot say?
Task 6. Condition on a second variable; Simpson’s paradox

Part A. Does the class pattern hold within each sex? This is a method chain: one decision per line. Complete the two blanks in .agg.

Part B: Simpson’s paradox. The Berkeley 1973 admissions data (six largest departments) is provided. Compare the overall rate with the within-department rate.

# Part A: survival by class AND sex (method chain, slide 45)
analysis = (
    df.loc[:, ["Survived", "Pclass", "Sex", "Fare"]]
      .query("Fare >= 0")
      .groupby(["Pclass", "Sex"], observed=True)
      .agg(passengers=("Survived", ___),          # how many rows?
           survival_rate=("Survived", ___),       # the proportion who survived
           median_fare=("Fare", "median"))
      .reset_index()
)
display(analysis.round({"survival_rate": 3, "median_fare": 2}))

male_share = df.groupby("Pclass")["Sex"].apply(lambda s: (s == "male").mean()).round(3)
print("share of men by class:", male_share.to_dict())

# Part B: Simpson's paradox (UC Berkeley 1973)
berk = pd.DataFrame({
    "dept":       list("ABCDEF"),
    "men_apps":   [825, 560, 325, 417, 191, 373],
    "men_adm":    [512, 353, 120, 138,  53,  22],
    "women_apps": [108,  25, 593, 375, 393, 341],
    "women_adm":  [ 89,  17, 202, 131,  94,  24],
})
berk["men_rate"]   = berk["men_adm"] / berk["men_apps"]
berk["women_rate"] = berk["women_adm"] / berk["women_apps"]

overall_men   = ___      # all men admitted / all men who applied
overall_women = ___
n_dept_women_higher = ___   # in how many departments is the women's rate higher?
print(f"overall: men {overall_men:.1%}, women {overall_women:.1%}")
print("departments where women's rate is higher:", n_dept_women_higher, "of 6")
berk.round(3)

Hint

check("first-class women: 94 passengers, 96.8% survived", analysis.query("Pclass == 1 and Sex == 'female'")[["passengers", "survival_rate"]].round(3).values.tolist() == [[94, 0.968]])
check("overall: men 44.5%, women 30.4%", abs(overall_men - 0.445) < 0.002 and abs(overall_women - 0.304) < 0.002)
check("women's rate is higher in 4 of 6 departments", n_dept_women_higher == 4)

Your answer (1 to 2 sentences). Overall, women were admitted less often, yet within most departments they were admitted more often. How can both be true?
Task 7. Split the data; leak-free scaling with NumPy

Rule: anything learned from the data (a median, a mean, a standard deviation) must be learned from the training rows only.

Part A. Make a stratified 80/20 split and compare train and test.

Part B. Standardise Age and Fare with NumPy broadcasting, using the training rows only, and apply the same numbers to the test rows.

# Part A: split
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=___, random_state=___, stratify=___      # 20% test, seed 42, stratify on the target
)
print(X_train.shape, X_test.shape)

def profile(Xf, yf):                                          # provided
    return pd.Series({"n": len(Xf), "survival": yf.mean(), "age_missing": Xf["Age"].isna().mean(),
                      "median_fare": Xf["Fare"].median(), "third_class": (Xf["Pclass"] == 3).mean()})
display(pd.DataFrame({"train": profile(X_train, y_train), "test": profile(X_test, y_test)}).T.round(3))

# Part B: leak-free standardisation
median_all = df["Age"].median()                               # FLAWED: this uses test rows too
median_train = ___                                            # the median Age of the TRAINING rows
A_train = X_train[["Age", "Fare"]].to_numpy(dtype=float)
A_test  = X_test[["Age", "Fare"]].to_numpy(dtype=float)
mu = ___                                                      # column means of A_train, ignoring NaN (np.nanmean, axis=0)
sd = ___                                                      # column standard deviations of A_train, ignoring NaN
Z_test = ___                                                  # standardise A_test with mu and sd
print("median Age: all rows", median_all, "| training rows only", median_train)
print("mu =", mu.round(2), "| sd =", sd.round(2))

Hint

check("split shapes (712, 7) and (179, 7)", X_train.shape == (712, 7) and X_test.shape == (179, 7))
check("the median comes from the training rows (28.5, not 28.0)", median_train == X_train["Age"].median() and median_train != median_all)
check("Z_test matches scikit-learn's StandardScaler fitted on the training rows",
      np.allclose(Z_test, StandardScaler().fit(A_train).transform(A_test), equal_nan=True))

Task 8. How big can a leak be?

Here the data are pure noise: 100 rows, 5,000 random features and random labels. There is no signal, so the honest accuracy is 50%.

You compare two ways of “choosing the 20 best features, then cross-validating”. Wrong: choose on all rows first. Right: choose inside a Pipeline, so each fold chooses from its own training rows only. Complete the four blanks (slide 41 shows the code).

def one_run(seed):
    rng = np.random.default_rng(seed)
    Xn = rng.normal(size=(100, 5000))        # 5,000 noise features
    yn = rng.integers(0, 2, 100)             # random labels
    clf = LogisticRegression(max_iter=1000)

    # WRONG: feature selection sees ALL the labels before cross-validation
    X_sel = SelectKBest(f_classif, k=20).fit_transform(___, ___)
    wrong = cross_val_score(clf, X_sel, yn, cv=5).mean()

    # RIGHT: selection happens inside each training fold
    pipe = Pipeline([("select", SelectKBest(f_classif, k=20)), ("clf", clf)])
    right = cross_val_score(___, ___, yn, cv=5).mean()
    return wrong, right

results = np.array([one_run(s) for s in range(20)])       # 20 random seeds (takes a few seconds)
wrong_mean, right_mean = results.mean(axis=0)
print(f"wrong (leaky): {wrong_mean:.3f}    right (pipeline): {right_mean:.3f}    (chance = 0.500)")

Hint

check("the leaky version scores far above chance (> 0.75) on pure noise", wrong_mean > 0.75)
check("the pipeline version is close to chance (0.45 to 0.55)", 0.45 < right_mean < 0.55)

Your answer (1 sentence). Why does the leaky version score so high on data with no signal?
Task 9. ColumnTransformer and Pipeline

Part A. Build one object that preprocesses numeric and categorical columns differently: numeric = median imputation + standard scaling; categorical = most-frequent imputation + one-hot encoding that ignores unseen categories. Fit it on the training rows only.

Part B. Attach a logistic regression, fit on the training rows, score on the test rows, and read the confusion matrix.

numeric = ["Age", "Fare", "SibSp", "Parch"]
categorical = ["Pclass", "Sex", "Embarked"]

# Part A
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy=___)),
                      ("scale",  ___)]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy=___)),
                      ("encode", OneHotEncoder(handle_unknown=___))]), categorical),
])
Xt = preprocess.fit_transform(X_train)                       # fit on TRAIN only
print(X_train.shape, "->", Xt.shape)

# Part B
model = Pipeline([("prep", ___),
                  ("clf", LogisticRegression(max_iter=1000))])
model.fit(___, ___)                                          # training data only
acc = model.score(___, ___)                                  # untouched test data
baseline_test = max(y_test.mean(), 1 - y_test.mean())        # provided
print(f"test accuracy {acc:.3f} vs majority baseline {baseline_test:.3f}")

pred = model.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["died", "survived"], digits=3))

Hint

check("7 columns in, 12 columns out", Xt.shape == (712, 12))
check("test accuracy is 0.804", abs(acc - 0.804) < 0.002)
check("23 survivors were missed (false negatives)", confusion_matrix(y_test, pred)[1, 0] == 23)

Your answer (1 to 2 sentences). The model beats the baseline. Why is accuracy alone not enough? Quote one number from the report.
Task 10. Cross-validation; write one finding

Part A. Pass the whole pipeline (and the raw X and y) to cross_val_score. Every fold refits every step, so nothing leaks.

Part B. The last step of an EDA: every finding becomes a decision with a limitation. Two rows are written for you. Write one more row, from anything you found today.

# Part A
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(___, ___, ___, cv=cv)       # which object? which data? which target?
print("fold accuracies:", scores.round(3))
print(f"mean {scores.mean():.3f}  +/- {scores.std():.3f}")

# Part B
report = pd.DataFrame([
    {"finding":    "Age is missing for 27.7% of 3rd class but 13.9% of 1st class",
     "decision":   "Impute with the training median inside the Pipeline, and keep a missing-indicator",
     "limitation": "MNAR cannot be excluded; imputation understates the variance"},
    {"finding":    "Fare is right-skewed (skew 4.79); 116 values above the IQR fence are mostly 1st class",
     "decision":   "Keep them; use log1p(Fare) or the median",
     "limitation": "Fare may be a ticket total, not a personal price"},
    {"finding": ___, "decision": ___, "limitation": ___},       # YOUR ROW: use anything you found today
])
report

Hint

check("mean cross-validated accuracy is 0.796", abs(scores.mean() - 0.796) < 0.002)
check("the spread across folds is about 0.033", abs(scores.std() - 0.033) < 0.003)
check("your finding row is filled in (a short sentence in each cell)", len(report) == 3 and report.astype(str).apply(lambda col: col.str.len().gt(10)).all().all())
