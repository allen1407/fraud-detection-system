"""Integrated fraud pipeline: validate -> engineer -> score -> explain -> trust -> route -> report."""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config as C
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FACTOR_LABELS = {
    "Transaction_Amount": "Transaction amount", "Customer_Age": "Customer age",
    "Customer_Income": "Customer income", "Account_Balance": "Account balance",
    "hour": "Hour of day", "day_of_week": "Day of week", "month": "Month", "is_weekend": "Weekend flag",
    "Amount_to_Income": "Amount / income", "Amount_to_Balance": "Amount / balance",
    "occupation_freq": "Occupation frequency", "merchant_freq": "Merchant frequency",
    "Transaction_Type": "Transaction type", "Category": "Merchant category",
    "Payment_Method": "Payment method", "Customer_Gender": "Gender",
}


class FraudPipeline:
    def __init__(self, models_dir=C.MODELS_DIR, build_explainer=True):
        import tensorflow as tf
        self.meta = json.loads((models_dir / "metadata.json").read_text())
        self.pre = joblib.load(models_dir / "preprocessor.pkl")
        self.lr = joblib.load(models_dir / "fraud_lr_model.pkl")
        self.dnn = tf.keras.models.load_model(models_dir / "fraud_dnn_model.keras")
        self.threshold = float(self.meta["threshold"])
        self.lr_threshold = float(self.meta["lr_threshold"])
        self.categories = self.meta["categories"]
        self.fe, self.ct = self.pre.named_steps["fe"], self.pre.named_steps["ct"]

        self.train = pd.read_csv(C.TRAIN_CSV, parse_dates=[C.DATE_COL])
        Xtr = self.pre.transform(self.train[C.INPUT_COLUMNS])
        self.nn = NearestNeighbors(n_neighbors=C.KNN_K + 1).fit(Xtr)
        loo = self.nn.kneighbors(Xtr)[0][:, 1:].mean(axis=1)           # leave-one-out kNN distance
        self.ref_distance = float(np.percentile(loo, 95))
        self._explainer = None
        self._bg = None
        if build_explainer:
            self._build_explainer()

    # ------------------------------------------------------------------ validation
    def validate(self, record: dict) -> list:
        errors = []
        for f in C.REQUIRED_FIELDS:
            v = record.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
                errors.append(f"Missing required field: {f}")
        if errors:
            return errors
        try:
            pd.to_datetime(record[C.DATE_COL])
        except Exception:
            errors.append(f"{C.DATE_COL} is not a valid date/time")
        for f in C.NUMERIC_INPUTS:
            try:
                v = float(record[f])
            except (TypeError, ValueError):
                errors.append(f"{f} must be a number")
                continue
            if not np.isfinite(v):
                errors.append(f"{f} must be a finite number")
            elif f == "Transaction_Amount" and v <= 0:
                errors.append("Transaction_Amount must be positive (negative or zero amounts are not allowed)")
            elif v < 0:
                errors.append(f"{f} must not be negative")
            else:
                lo, hi = C.RANGES[f]
                if not lo <= v <= hi:
                    errors.append(f"{f}={v:g} is outside the valid range [{lo:g}, {hi:g}]")
        for f in C.CATEGORICAL_INPUTS:
            if str(record[f]) not in self.categories[f]:
                errors.append(f"Unknown {f}: '{record[f]}' (allowed: {', '.join(self.categories[f])})")
        return errors

    # ------------------------------------------------------------------ scoring
    def _frame(self, records) -> pd.DataFrame:
        df = pd.DataFrame(records)[C.INPUT_COLUMNS].copy()
        for f in C.NUMERIC_INPUTS:
            df[f] = df[f].astype(float)
        df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL])
        return df

    def score(self, records) -> pd.DataFrame:
        """records: dict, list of dicts or DataFrame of VALID rows -> DNN and LR probabilities."""
        if isinstance(records, dict):
            records = [records]
        df = self._frame(records)
        X = self.pre.transform(df)
        dnn_p = self.dnn(X.astype("float32"), training=False).numpy().ravel()
        lr_p = self.lr.predict_proba(X)[:, 1]
        return pd.DataFrame({"dnn_prob": dnn_p, "lr_prob": lr_p, "_X": list(X)}, index=range(len(df)))

    def trust_scores(self, X) -> np.ndarray:
        d = self.nn.kneighbors(np.asarray(X), n_neighbors=C.KNN_K)[0].mean(axis=1)
        return np.minimum(self.ref_distance / np.maximum(d, 1e-9), 10.0)

    def trust_score(self, record: dict) -> float:
        X = self.pre.transform(self._frame([record]))
        return float(self.trust_scores(X)[0])

    def route(self, dnn_prob, lr_prob, trust) -> str:
        dnn_flag, lr_flag = dnn_prob >= self.threshold, lr_prob >= self.lr_threshold
        if trust >= C.TRUST_MIN and dnn_flag and lr_flag:
            return "AUTO-BLOCK"
        if trust >= C.TRUST_MIN and not dnn_flag and not lr_flag:
            return "AUTO-APPROVE"
        return "MANUAL REVIEW"

    # ------------------------------------------------------------------ explainability
    def _decode(self, M):
        M = np.atleast_2d(M)
        n_num = len(NUMERIC_FEATURES)
        df = pd.DataFrame(M[:, :n_num].astype(float), columns=NUMERIC_FEATURES)
        for j, c in enumerate(CATEGORICAL_FEATURES):
            cats = np.array(self.categories[c])
            idx = np.clip(np.rint(M[:, n_num + j]).astype(int), 0, len(cats) - 1)
            df[c] = cats[idx]
        return df

    def _encode(self, fe_df: pd.DataFrame) -> np.ndarray:
        M = fe_df[NUMERIC_FEATURES].to_numpy(float)
        codes = np.column_stack([[self.categories[c].index(v) for v in fe_df[c]] for c in CATEGORICAL_FEATURES])
        return np.hstack([M, codes])

    def _model_fn(self, M):
        X = self.ct.transform(self._decode(M)).astype("float32")
        return self.dnn(X, training=False).numpy().ravel()

    def _build_explainer(self):
        import shap
        bg = self.train.sample(100, random_state=C.SEED)
        self._bg = self._encode(self.fe.transform(bg[C.INPUT_COLUMNS]))
        self._explainer = shap.KernelExplainer(self._model_fn, self._bg)

    def shap_values(self, fe_matrix, nsamples=200):
        return np.asarray(self._explainer.shap_values(fe_matrix, nsamples=nsamples, silent=True))

    def explain(self, record: dict, top_k=5) -> list:
        fe_df = self.fe.transform(self._frame([record]))
        sv = self.shap_values(self._encode(fe_df))[0]
        cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        out = []
        for i in np.argsort(-np.abs(sv))[:top_k]:
            v = fe_df.iloc[0][cols[i]]
            out.append(dict(feature=cols[i], label=FACTOR_LABELS[cols[i]],
                            value=v if isinstance(v, str) else float(v), shap=float(sv[i]),
                            direction="increases fraud risk" if sv[i] > 0 else "decreases fraud risk"))
        return out

    # ------------------------------------------------------------------ orchestration
    def assess(self, record: dict, explain=True) -> dict:
        errors = self.validate(record)
        if errors:
            return dict(valid=False, errors=errors, verdict="TRANSACTION REJECTED",
                        recommendation=C.RECOMMENDATIONS["TRANSACTION REJECTED"], record=record)
        s = self.score(record).iloc[0]
        trust = float(self.trust_scores(s["_X"][None, :])[0])
        verdict = self.route(s.dnn_prob, s.lr_prob, trust)
        dnn_flag, lr_flag = s.dnn_prob >= self.threshold, s.lr_prob >= self.lr_threshold
        return dict(valid=True, errors=[], record=record, verdict=verdict,
                    dnn_prob=float(s.dnn_prob), lr_prob=float(s.lr_prob),
                    dnn_flag=bool(dnn_flag), lr_flag=bool(lr_flag), models_agree=bool(dnn_flag == lr_flag),
                    trust=trust, recommendation=C.RECOMMENDATIONS[verdict],
                    factors=self.explain(record) if explain else [])

    def generate_report(self, result: dict) -> str:
        L = ["=" * 60, "FRAUD DETECTION REPORT", "=" * 60]
        if not result["valid"]:
            L += ["VERDICT: TRANSACTION REJECTED", "", "Reasons:"] + [f"  - {e}" for e in result["errors"]]
            L += ["", f"Recommendation: {result['recommendation']}"]
            return "\n".join(L)
        L += [f"VERDICT: {result['verdict']}", "",
              f"DNN fraud probability : {result['dnn_prob']:.3f} (threshold {self.threshold:.2f}) -> "
              f"{'FLAG' if result['dnn_flag'] else 'OK'}",
              f"LR  fraud probability : {result['lr_prob']:.3f} (threshold {self.lr_threshold:.2f}) -> "
              f"{'FLAG' if result['lr_flag'] else 'OK'}",
              f"Model agreement       : {'AGREE' if result['models_agree'] else 'DISAGREE'}",
              f"Trust score           : {result['trust']:.2f} (>= {C.TRUST_MIN:.1f} required for automatic action)",
              "", f"Recommendation: {result['recommendation']}", "", "Top factors (SHAP, DNN):"]
        for i, f in enumerate(result["factors"], 1):
            val = f["value"] if isinstance(f["value"], str) else f"{f['value']:.4g}"
            L.append(f"  {i}. {f['label']} = {val}: {f['direction']} ({f['shap']:+.4f})")
        L += ["", "Note: decision support only; final decisions on MANUAL REVIEW cases rest with a human analyst."]
        return "\n".join(L)

    def score_batch(self, df: pd.DataFrame, explain=False) -> pd.DataFrame:
        records = df.to_dict("records")
        errs = []
        for r in records:
            try:
                errs.append(self.validate(r))
            except Exception as e:                       # never crash a batch
                errs.append([f"Row could not be validated: {e}"])
        out = pd.DataFrame(index=df.index, columns=["verdict", "dnn_prob", "lr_prob", "models_agree",
                                                    "trust", "recommendation", "errors", "top_factors"], dtype=object)
        ok = [i for i, e in enumerate(errs) if not e]
        for i, e in enumerate(errs):
            if e:
                out.iloc[i] = ["TRANSACTION REJECTED", np.nan, np.nan, np.nan, np.nan,
                               C.RECOMMENDATIONS["TRANSACTION REJECTED"], "; ".join(e), ""]
        if ok:
            s = self.score([records[i] for i in ok])
            X = np.vstack(s["_X"].to_numpy())
            trust = self.trust_scores(X)
            for k, i in enumerate(ok):
                v = self.route(s.dnn_prob[k], s.lr_prob[k], trust[k])
                dflag, lflag = s.dnn_prob[k] >= self.threshold, s.lr_prob[k] >= self.lr_threshold
                factors = ""
                if explain:
                    factors = "; ".join(f"{f['label']} ({f['shap']:+.3f})" for f in self.explain(records[i]))
                out.iloc[i] = [v, float(s.dnn_prob[k]), float(s.lr_prob[k]), bool(dflag == lflag),
                               float(trust[k]), C.RECOMMENDATIONS[v], "", factors]
        return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
