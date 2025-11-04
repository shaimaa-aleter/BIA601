import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    RFE,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .config import CFG
from .utils import smart_k, evaluate_subset


def traditional_selectors_report(X_train_df, y_train, X_test_df, y_test, k_mode: str = None, base_estimator=None) -> pd.DataFrame:
    if base_estimator is None:
        base_estimator = RandomForestClassifier(
            n_estimators=CFG.RF_N_EST, random_state=CFG.RANDOM_STATE, n_jobs=-1
        )

    n_feats = X_train_df.shape[1]
    k_eff = smart_k(n_feats) if (k_mode is None or k_mode == "auto") else max(1, min(CFG.TOP_K, n_feats))

    methods = OrderedDict()

    vt = VarianceThreshold()
    X_tr_vt = vt.fit_transform(X_train_df)
    X_te_vt = vt.transform(X_test_df)
    feats_vt = X_train_df.columns[vt.get_support()].tolist()
    methods["Variance Threshold"] = (X_tr_vt, X_te_vt, feats_vt)

    skb = SelectKBest(f_classif, k=min(k_eff, X_train_df.shape[1]))
    X_tr_skb = skb.fit_transform(X_train_df, y_train)
    X_te_skb = skb.transform(X_test_df)
    feats_skb = X_train_df.columns[skb.get_support()].tolist()
    methods["SelectKBest (ANOVA)"] = (X_tr_skb, X_te_skb, feats_skb)

    mm = MinMaxScaler()
    X_tr_mm = mm.fit_transform(X_train_df)
    X_te_mm = mm.transform(X_test_df)
    chi = SelectKBest(chi2, k=min(k_eff, X_train_df.shape[1]))
    X_tr_chi = chi.fit_transform(X_tr_mm, y_train)
    X_te_chi = chi.transform(X_te_mm)
    feats_chi = X_train_df.columns[chi.get_support()].tolist()
    methods["Chi-Square (χ²)"] = (X_tr_chi, X_te_chi, feats_chi)

    mi = SelectKBest(mutual_info_classif, k=min(k_eff, X_train_df.shape[1]))
    X_tr_mi = mi.fit_transform(X_train_df, y_train)
    X_te_mi = mi.transform(X_test_df)
    feats_mi = X_train_df.columns[mi.get_support()].tolist()
    methods["Mutual Information"] = (X_tr_mi, X_te_mi, feats_mi)

    lr = LogisticRegression(max_iter=3000, solver="lbfgs", n_jobs=-1)
    rfe = RFE(lr, n_features_to_select=min(k_eff, X_train_df.shape[1]), step=1)
    X_tr_rfe = rfe.fit_transform(X_train_df, y_train)
    X_te_rfe = rfe.transform(X_test_df)
    feats_rfe = X_train_df.columns[rfe.get_support()].tolist()
    methods["RFE (Logistic Regression)"] = (X_tr_rfe, X_te_rfe, feats_rfe)

    rf_tmp = RandomForestClassifier(n_estimators=200, random_state=CFG.RANDOM_STATE, n_jobs=-1)
    rf_tmp.fit(X_train_df, y_train)
    importances = rf_tmp.feature_importances_
    idx = np.argsort(importances)[::-1][: min(k_eff, len(importances))]
    feats_rf = X_train_df.columns[idx].tolist()
    methods["RandomForest Importance"] = (
        X_train_df[feats_rf].values,
        X_test_df[feats_rf].values,
        feats_rf,
    )

    rows = []
    for name, (Xtr, Xte, feats) in methods.items():
        metrics = evaluate_subset(base_estimator, Xtr, y_train, Xte, y_test, use_proba=True)
        rows.append(
            dict(
                Method=name,
                Accuracy=metrics["accuracy"],
                ROC_AUC=metrics["roc_auc"],
                PR_AUC=metrics["pr_auc"],
                Selected=len(feats),
            )
        )

    return pd.DataFrame(rows).sort_values(by=["ROC_AUC", "PR_AUC"], ascending=False).reset_index(drop=True)