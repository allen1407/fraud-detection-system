"""Train logistic regression + Keras DNN, choose thresholds, run the sanity gate."""
import json
import os
import random
import sys
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import config as C
from .data import load_clean, split_and_save
from .features import build_preprocessor, feature_names


def set_seeds():
    import tensorflow as tf
    random.seed(C.SEED); np.random.seed(C.SEED); tf.keras.utils.set_random_seed(C.SEED)


def build_dnn(n_features: int):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input((n_features,)),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m


def cm_counts(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def cost_at(y, p, t):
    _, fp, fn, _ = cm_counts(y, (p >= t).astype(int))
    return fn * C.COST_FN + fp * C.COST_FP


def threshold_sweep(y, p, grid=None):
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2) if grid is None else grid
    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = cm_counts(y, pred)
        rows.append(dict(threshold=float(t), f1=f1_score(y, pred, zero_division=0),
                         flag_rate=float(pred.mean()), fpr=fp / max(fp + tn, 1),
                         cost=fn * C.COST_FN + fp * C.COST_FP))
    return rows


def choose_thresholds(y, p):
    rows = threshold_sweep(y, p)
    f1_opt = max(rows, key=lambda r: r["f1"])
    guarded = [r for r in rows if r["fpr"] <= C.MAX_VAL_FPR]
    f1_guard = max(guarded, key=lambda r: r["f1"]) if guarded else min(rows, key=lambda r: r["fpr"])
    cost_opt = min(rows, key=lambda r: r["cost"])
    return f1_opt, f1_guard, cost_opt


def test_report(name, y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = cm_counts(y, pred)
    r = dict(model=name, threshold=float(t), roc_auc=roc_auc_score(y, p),
             pr_auc=average_precision_score(y, p), f1=f1_score(y, pred),
             precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
             flag_rate=float(pred.mean()), fpr=fp / max(fp + tn, 1), tn=tn, fp=fp, fn=fn, tp=tp)
    print(f"{name}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items() if k != "model"))
    return r


def main():
    import tensorflow as tf
    set_seeds()
    C.MODELS_DIR.mkdir(exist_ok=True)
    df = load_clean()
    train, val, test = split_and_save(df)
    print(f"rows: train={len(train)} val={len(val)} test={len(test)}; fraud rate train={train[C.TARGET].mean():.4f}")

    pre = build_preprocessor().fit(train[C.INPUT_COLUMNS])        # fit ONLY on train
    Xtr, Xva, Xte = (pre.transform(d[C.INPUT_COLUMNS]) for d in (train, val, test))
    ytr, yva, yte = (d[C.TARGET].values for d in (train, val, test))
    print("preprocessed shape:", Xtr.shape, "NaNs:", int(np.isnan(Xtr).sum()))

    lr = LogisticRegression(max_iter=1000, random_state=C.SEED).fit(Xtr, ytr)

    dnn = build_dnn(Xtr.shape[1])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    # ~200k rows: a larger batch keeps an epoch to a few seconds; imbalance is handled by threshold choice, not weights
    hist = dnn.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=100, batch_size=512, callbacks=[es], verbose=0)
    print("DNN epochs run:", len(hist.history["loss"]), "best val_loss:", round(min(hist.history["val_loss"]), 4))

    pva = dnn.predict(Xva, verbose=0).ravel()
    pte = dnn.predict(Xte, verbose=0).ravel()
    f1_opt, f1_guard, cost_opt = choose_thresholds(yva, pva)
    print("val F1-optimal (unconstrained):", f1_opt)
    print(f"val F1-optimal with FPR<={C.MAX_VAL_FPR} guardrail:", f1_guard)
    print("val cost-optimal:", cost_opt)
    thr = f1_guard["threshold"]

    # ---- SANITY GATE ----
    print("\n=== SANITY GATE (test set) ===")
    rd = test_report("DNN", yte, pte, thr)
    rl = test_report("LR ", yte, lr.predict_proba(Xte)[:, 1], C.LR_THRESHOLD)
    ok = True
    for r in (rd, rl):
        if r["flag_rate"] > 0.90 or r["fpr"] > 0.50:
            ok = False
            print(f"GATE FAIL for {r['model']}: flag_rate={r['flag_rate']:.2f} fpr={r['fpr']:.2f}")
    if rd["roc_auc"] < 0.6:
        print(f"WARNING: weak signal, DNN test ROC-AUC={rd['roc_auc']:.3f} < 0.6 (dataset has little predictive information).")

    joblib.dump(pre, C.PREPROCESSOR_PATH)
    joblib.dump(lr, C.LR_PATH)
    dnn.save(C.DNN_PATH)
    meta = dict(
        feature_list=feature_names(pre), input_columns=C.INPUT_COLUMNS,
        threshold=thr, lr_threshold=C.LR_THRESHOLD,
        threshold_f1_unconstrained=f1_opt["threshold"],
        threshold_f1_guardrail=f1_guard["threshold"], max_val_fpr_guardrail=C.MAX_VAL_FPR,
        threshold_cost_optimal=cost_opt["threshold"], cost_fn=C.COST_FN, cost_fp=C.COST_FP,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset_rows=int(len(df)), split_sizes=dict(train=len(train), val=len(val), test=len(test)),
        epochs_run=len(hist.history["loss"]),
        library_versions=dict(python=sys.version.split()[0], numpy=np.__version__, sklearn=sklearn.__version__,
                              tensorflow=tf.__version__, pandas=__import__("pandas").__version__),
    )
    C.METADATA_PATH.write_text(json.dumps(meta, indent=2))
    print("\nSaved models + metadata. Sanity gate:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
