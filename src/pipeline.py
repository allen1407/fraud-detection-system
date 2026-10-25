"""Integrated fraud pipeline: validate -> engineer -> score -> explain -> trust -> route -> report."""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config as C
from .features import NUMERIC_FEATURES

FACTOR_LABELS = {
    **{v: f"{v} (PCA component)" for v in C.PCA_INPUTS},
    "Amount": "Transaction amount", "hour": "Hour of day", "is_night": "Night-time flag (00-06 h)",
    "log_amount": "log(1 + amount)",
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
        self.fe, self.ct = self.pre.named_steps["fe"], self.pre.named_steps["ct"]

        self.train = pd.read_csv(C.TRAIN_CSV)
        # reference set: every training fraud + a fixed sample of legitimate rows (keeps the kNN index small)
        legit = self.train[self.train[C.TARGET] == 0]
        self.reference = pd.concat([self.train[self.train[C.TARGET] == 1],
                                    legit.sample(min(C.TRUST_REF_LEGIT, len(legit)), random_state=C.SEED)])
        Xref = self.pre.transform(self.reference[C.INPUT_COLUMNS])
        self.nn = NearestNeighbors(n_neighbors=C.KNN_K + 1).fit(Xref)
        loo = self.nn.kneighbors(Xref)[0][:, 1:].mean(axis=1)          # leave-one-out kNN distance
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
        for f in C.NUMERIC_INPUTS:
            try:
                v = float(record[f])
            except (TypeError, ValueError):
                errors.append(f"{f} must be a number")
                continue
            if not np.isfinite(v):
                errors.append(f"{f} must be a finite number")
            elif f in ("Amount", C.TIME_COL) and v < 0:
                errors.append(f"{f} must not be negative")
            else:
                lo, hi = C.RANGES[f]
                if not lo <= v <= hi:
                    errors.append(f"{f}={v:g} is outside the valid range [{lo:g}, {hi:g}]")
        return errors

    # ------------------------------------------------------------------ scoring
    def _frame(self, records) -> pd.DataFrame:
        df = pd.DataFrame(records)[C.INPUT_COLUMNS].copy()
        return df.astype(float)

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
    def _model_fn(self, M):
        X = self.ct.transform(pd.DataFrame(np.atleast_2d(M), columns=NUMERIC_FEATURES)).astype("float32")
        return self.dnn(X, training=False).numpy().ravel()

    def _build_explainer(self):
        import shap
        bg = self.train.sample(100, random_state=C.SEED)
        self._bg = self.fe.transform(bg[C.INPUT_COLUMNS]).to_numpy(float)
        self._explainer = shap.KernelExplainer(self._model_fn, self._bg)

    def shap_values(self, fe_matrix, nsamples=200):
        return np.asarray(self._explainer.shap_values(fe_matrix, nsamples=nsamples, silent=True))

    def explain(self, record: dict, top_k=5) -> list:
        fe_df = self.fe.transform(self._frame([record]))
        sv = self.shap_values(fe_df.to_numpy(float))[0]
        cols = NUMERIC_FEATURES
        out = []
        for i in np.argsort(-np.abs(sv))[:top_k]:
            out.append(dict(feature=cols[i], label=FACTOR_LABELS[cols[i]],
                            value=float(fe_df.iloc[0][cols[i]]), shap=float(sv[i]),
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
            L.append(f"  {i}. {f['label']} = {f['value']:.4g}: {f['direction']} ({f['shap']:+.4f})")
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
                    factors = "; ".join(f"{f['label']} ({f['shap']:+.3g})" for f in self.explain(records[i]))
                out.iloc[i] = [v, float(s.dnn_prob[k]), float(s.lr_prob[k]), bool(dflag == lflag),
                               float(trust[k]), C.RECOMMENDATIONS[v], "", factors]
        return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
