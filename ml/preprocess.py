import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .config import CFG
from .utils import iqr_cap_series


def preprocess(df: pd.DataFrame, target_col: str, clip_outliers: bool = True):
    assert target_col in df.columns, f"target_col '{target_col}' not in dataframe"
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in df.columns if c not in categorical_cols + [target_col]]

    df_capped = df.copy()
    if clip_outliers and len(numeric_cols) > 0:
        for col in numeric_cols:
            df_capped[col] = iqr_cap_series(df_capped[col], k=1.5)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])

    X = df_capped.drop(columns=[target_col])
    y = df_capped[target_col].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CFG.TEST_SIZE,
        random_state=CFG.RANDOM_STATE,
        stratify=y if CFG.STRATIFY and len(np.unique(y)) == 2 else None,
    )

    X_train_np = pre.fit_transform(X_train)
    X_test_np = pre.transform(X_test)

    num_names = numeric_cols
    ohe_names = []
    if len(categorical_cols) > 0:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
    feature_names = num_names + ohe_names

    X_train_df = pd.DataFrame(X_train_np, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_np, columns=feature_names, index=X_test.index)

    n_nans = X_train_df.isna().sum().sum()
    n_inf = np.isinf(X_train_df.to_numpy()).sum()
    assert n_nans == 0 and n_inf == 0, "NaN/Inf found after preprocessing!"
    assert all(np.issubdtype(dt, np.number) for dt in X_train_df.dtypes), "Non-numeric feature remained!"

    return X_train_df, X_test_df, y_train, y_test, feature_names