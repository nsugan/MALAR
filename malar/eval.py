"""Eval harness — sensitivity / specificity / F1 / AUROC on the Raman demo.

Trains the learned spectral representation on a labelled Raman training split, forms
per-class prototypes in latent space (the matured f_k taking over from the LLM), then
evaluates identification on a held-out split. Emits a JSON report.

Usage: python -m malar.eval [--domain raman_virus] [--out data/eval_report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from malar.encoders.spectral import SpectralEncoder
from malar.world.adapters.raman import RamanAdapter


def _gather(adapter: RamanAdapter, n_ticks: int):
    X, y = [], []
    for i, b in enumerate(adapter.stream()):
        if i >= n_ticks:
            break
        X.append(b.points)
        y.extend(b.labels)
    return np.vstack(X), np.array(y)


def _prototypes(Z: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {c: Z[y == c].mean(axis=0) for c in np.unique(y)}


def _predict(Z: np.ndarray, protos: dict[str, np.ndarray]):
    classes = list(protos)
    P = np.stack([protos[c] for c in classes])
    # negative euclidean distance -> softmax scores
    d = np.linalg.norm(Z[:, None, :] - P[None, :, :], axis=2)
    scores = np.exp(-d)
    scores = scores / scores.sum(axis=1, keepdims=True)
    pred = np.array([classes[i] for i in scores.argmax(axis=1)])
    return pred, scores, classes


def evaluate(seed: int = 0) -> dict:
    train_ad = RamanAdapter(n_ticks=4, n_per_class=10, n_bands=128, seed=seed)
    test_ad = RamanAdapter(n_ticks=2, n_per_class=8, n_bands=128, seed=seed + 99)
    Xtr, ytr = _gather(train_ad, 4)
    Xte, yte = _gather(test_ad, 2)

    enc = SpectralEncoder(latent_dim=16).fit(Xtr)
    Ztr = enc.encode_matrix(Xtr)
    Zte = enc.encode_matrix(Xte)
    protos = _prototypes(Ztr, ytr)
    pred, scores, classes = _predict(Zte, protos)

    report = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "classes": classes,
              "per_class": {}, "overall": {}}
    f1s = []
    for ci, c in enumerate(classes):
        tp = int(np.sum((pred == c) & (yte == c)))
        fp = int(np.sum((pred == c) & (yte != c)))
        fn = int(np.sum((pred != c) & (yte == c)))
        tn = int(np.sum((pred != c) & (yte != c)))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        f1s.append(f1)
        auroc = _auroc((yte == c).astype(int), scores[:, ci])
        report["per_class"][c] = {"sensitivity": round(sens, 3), "specificity": round(spec, 3),
                                  "precision": round(prec, 3), "f1": round(f1, 3),
                                  "auroc": round(auroc, 3)}
    acc = float(np.mean(pred == yte))
    report["overall"] = {"accuracy": round(acc, 3), "macro_f1": round(float(np.mean(f1s)), 3),
                         "recon_error": round(enc.recon_error(Xte), 4)}
    return report


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="raman_virus")
    ap.add_argument("--out", default="data/eval_report.json")
    args = ap.parse_args(argv)
    rep = evaluate()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nReport written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
