import pandas as pd

from bayesian_optimization import BayesianOptimization

# Load the dataset
df = pd.read_excel('finaccess2024_datasprint.xlsx')

df.fillna(value={"barriers_bank": "No barrier"}, inplace=True)


# Handle the education_level column
education_col = df['education_level']
num = pd.to_numeric(education_col, errors='coerce')
is_int = num.notna() & (num % 1 == 0)
df = df.loc[~is_int]


# Handle the barriers_mobile_money column
df.loc[df['barriers_mobile_money'] == 0, 'barriers_mobile_money'] = 'No barrier'

invalid_rows = df[
    (df['marital_status'].isin(["Don't know   (DO NOT READ OUT)", "Refused to Answer(DO NOT READ OUT)"])) 
    | (df['education_level'] == "\"Refused to Answer (DO NOT READ OUT)\"")
    ].index
df.drop(invalid_rows, inplace=True)


one_hot_cols = [
    'location_type',
    'Sex',
    'Age',
    'education_level',
    'marital_status',
    'Savings_formal',
    'Savings_informal',
    'Loan_formal',
    'Loan_informal',
    'defaulted',
    'formal_service_use',
    'mobile_money_access',
    'mobile_ownership_1',
    'experienced_shock',
    'nfhi_11',
    'nfhi_12',
    'nfhi_13',
    'accessto_13k_1month',
    'not_difficult',
    'has_disability'
]
"""Columns to be one-hot encoded"""

target_cols = ['county', 'barriers_mobile_money',
    'barriers_bank']
"""Columns to be encoded using TargetEncoder"""

num_cols = ['household_size', 'monthly_income']
"""Columns to be scaled using StandardScaler"""


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, OrdinalEncoder, StandardScaler, LabelEncoder

RANDOM_STATE = 42
"""Set the global random state in order to get consistent results on each re-run"""

le = LabelEncoder()
y_encoded = le.fit_transform(df['financial_status'])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), one_hot_cols),
        ("ord", OrdinalEncoder(categories=[['None correct', 'One correct', 'Two correct', 'All correct']]), ['fl_score']),
        ("num", StandardScaler(), num_cols),
        ("county_te", TargetEncoder(smooth="auto", cv=5, random_state=RANDOM_STATE), target_cols)
    ],
    remainder="drop"
)

data = preprocessor.fit_transform(df.drop(columns=["financial_status"]), y_encoded)

from sklearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import loguniform

bo = BayesianOptimization(
    SVC(kernel="rbf", random_state=RANDOM_STATE),
    param_distributions= {
        "C": loguniform(0.01, 100),
        "gamma": loguniform(0.0001, 1)
    },
    n_iter=1,
    n_initial=2,
    cv=2,
    n_candidates=10_000,
    logging=True
)

# Create the pipeline model
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    # ('feature_selection', feature_selection),
    ('bo', bo)
])

# Properly split the dataset from the dataframe
X = df.drop(columns=['financial_status'])
y = le.fit_transform(df['financial_status'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

print("Fitting pipeline...\n\n")
pipeline.fit(X_train, y_train)
print("Fitting complete!\n")

from sklearn.metrics import accuracy_score, classification_report, f1_score

y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred, average="weighted")}")

labels = le.inverse_transform([0, 1, 2])

print(classification_report(y_test, y_pred, target_names=labels))