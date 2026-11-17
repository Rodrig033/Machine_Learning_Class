import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from yeast_data import load_yeast, stratified_split
from yeast_nn import BASE_CFG, train_model

# ERL tiene solo 5 muestras: sklearn avisa que tiene menos miembros que pliegues
warnings.filterwarnings("ignore", message="The least populated class")

SEED = 42
UCI_BASELINE = 0.5687
INNER_VAL_FRAC = 0.15

NAME_A = "A: red individual (parada temprana)"
NAME_B = "B: red individual (épocas fijas)"
NAME_C = "C: ensamble (parada temprana)"
NAME_D = "D: ensamble (épocas fijas)"

REFERENCE = {
    "Random Forest (sin ajustar)": lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    "SVM RBF (sin ajustar)": lambda: make_pipeline(StandardScaler(), SVC()),
    "Reg. logística (sin ajustar)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "k-vecinos k=15 (sin ajustar)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
}


# Utilidades
def holdout_split(y, frac, seed=0):
    """Separa una fracción de cada clase para validación interna (estratificado)."""
    rng = np.random.default_rng(seed)
    fit, val = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = 0 if len(idx) == 1 else max(1, int(round(len(idx) * frac)))
        val.extend(idx[:n_val])
        fit.extend(idx[n_val:])
    return np.array(fit), np.array(val)


def tt(X, y=None):
    Xt = torch.tensor(X, dtype=torch.float32)
    return Xt if y is None else (Xt, torch.tensor(y, dtype=torch.long))


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.softmax(model(X), dim=1).numpy()


def acc(probs, y):
    return float(np.mean(probs.argmax(axis=1) == y))


def train_nets(X_tr, y_tr, X_te, cfg, seeds, n_classes, inner_val_frac):
    """Entrena una red por semilla y devuelve sus probabilidades sobre X_te.

    El escalado se ajusta solo con X_tr. Si inner_val_frac es None, se entrena con todo
    X_tr y NO se usa ninguna validación (épocas fijas)."""
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    if inner_val_frac:
        i_fit, i_val = holdout_split(y_tr, inner_val_frac)
        train, val = (X_tr_s[i_fit], y_tr[i_fit]), (X_tr_s[i_val], y_tr[i_val])
    else:
        # con patience=None y restore_best=False la "validación" solo se registra, no decide nada
        train, val = (X_tr_s, y_tr), (X_tr_s[:64], y_tr[:64])

    data = dict(train=tt(*train), val=tt(*val), n_classes=n_classes)
    X_te_t = tt(X_te_s)
    probs, best_eps = [], []
    for s in seeds:
        model, _, best_ep = train_model(data, cfg, s)
        probs.append(predict_proba(model, X_te_t))
        best_eps.append(best_ep)
    return probs, best_eps


def nn_cv(folds, X, y, cfg, seeds, n_classes, inner_val_frac, label):
    single, ens, best_eps = [], [], []
    for k, (tr, te) in enumerate(folds, 1):
        probs, eps = train_nets(X[tr], y[tr], X[te], cfg, seeds, n_classes, inner_val_frac)
        single.append(np.mean([acc(p, y[te]) for p in probs]))
        ens.append(acc(np.mean(probs, axis=0), y[te]))
        best_eps += eps
        print(f"  [{label}] pliegue {k:2d}/{len(folds)}: red individual {single[-1]:.3f} | ensamble {ens[-1]:.3f}")
    return single, ens, best_eps


def reference_cv(folds, X, y):
    out = {name: [] for name in REFERENCE}
    for k, (tr, te) in enumerate(folds, 1):
        for name, make in REFERENCE.items():
            out[name].append(make().fit(X[tr], y[tr]).score(X[te], y[te]))
        print(f"  [referencia] pliegue {k:2d}/{len(folds)}: "
              + " | ".join(f"{n.split(' (')[0]} {v[-1]:.3f}" for n, v in out.items()))
    return out


def summarize(results, n_tr, n_te, baseline_name):
    """Resumen con prueba t remuestreada corregida (Nadeau y Bengio) frente a la variante A."""
    base = np.array(results[baseline_name])
    rows = []
    for name, vals in results.items():
        v = np.array(vals)
        d = v - base
        k = len(d)
        if name == baseline_name or d.var(ddof=1) == 0:
            p = np.nan
        else:
            t = d.mean() / np.sqrt((1 / k + n_te / n_tr) * d.var(ddof=1))
            p = 2 * stats.t.sf(abs(t), k - 1)
        rows.append({"método": name, "acc_media": v.mean(), "desv_pliegues": v.std(ddof=1),
                     "dif_vs_A": d.mean(), "p_corregido": p})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", default=None, help="ruta de yeast.data (opcional)")
    parser.add_argument("--config", default="mejor_configuracion.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--nets", type=int, default=5, help="redes por ensamble")
    parser.add_argument("--reference", action="store_true", help="incluir modelos de referencia")
    parser.add_argument("--quick", action="store_true", help="3 pliegues, 1 repetición, 2 redes")
    args = parser.parse_args()
    if args.quick:
        args.folds, args.repeats, args.nets = 3, 1, 2

    # --- Datos: misma partición de siempre; aquí solo se usa entrenamiento + validación
    X, y = load_yeast(args.local)
    y_enc = LabelEncoder().fit_transform(y)
    n_classes = len(np.unique(y_enc))
    i_tr, i_va, i_te = stratified_split(y_enc, SEED)
    i_dev = np.concatenate([i_tr, i_va])
    X_np = X.values
    X_dev, y_dev = X_np[i_dev], y_enc[i_dev]
    X_test, y_test = X_np[i_te], y_enc[i_te]
    print(f"Desarrollo (entrenamiento+validación): {len(i_dev)} muestras | Prueba reservada: {len(i_te)}")

    # --- Configuración de la red
    prev_test = None
    if os.path.exists(args.config):
        with open(args.config) as f:
            saved = json.load(f)
        cfg_base, prev_test = saved["config"], saved.get("acc_test_media")
        print(f"Configuración cargada de {args.config}")
    else:
        cfg_base = {**BASE_CFG, "n_layers": 3, "n_neurons": 16, "activation": "tanh",
                    "lr": 0.01, "dropout": 0.2}
        print("No se encontró el JSON: se usa la mejor configuración de la búsqueda anterior")
    print("  ", json.dumps(cfg_base))

    seeds = list(range(args.nets))
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=SEED)
    folds = list(cv.split(X_dev, y_dev))
    print(f"\nValidación cruzada: {args.folds} pliegues x {args.repeats} repeticiones = {len(folds)} evaluaciones, "
          f"{args.nets} redes por ensamble")

    # --- Pasada 1: parada temprana (A y C); de aquí sale el número de épocas para B y D
    print("\n--- Variantes con parada temprana (A, C) ---")
    cfg_early = {**cfg_base, "epochs": 300, "patience": 30, "restore_best": True}
    a_single, c_ens, best_eps = nn_cv(folds, X_dev, y_dev, cfg_early, seeds, n_classes, INNER_VAL_FRAC, "early")
    fixed_epochs = max(1, int(np.median(best_eps)))
    print(f"\nÉpocas fijas para B y D = mediana de las mejores épocas = {fixed_epochs}")

    # --- Pasada 2: épocas fijas, sin validación interna (B y D)
    print("\n--- Variantes con épocas fijas (B, D) ---")
    cfg_fixed = {**cfg_base, "epochs": fixed_epochs, "patience": None, "restore_best": False}
    b_single, d_ens, _ = nn_cv(folds, X_dev, y_dev, cfg_fixed, seeds, n_classes, None, "fijas")

    results = {NAME_A: a_single, NAME_B: b_single, NAME_C: c_ens, NAME_D: d_ens}

    if args.reference:
        print("\n--- Modelos de referencia (sin ajustar hiperparámetros) ---")
        results.update(reference_cv(folds, X_dev, y_dev))

    # --- Resultados de la validación cruzada
    n_tr, n_te = len(folds[0][0]), len(folds[0][1])
    summary = summarize(results, n_tr, n_te, NAME_A)
    print("\n=== Resultados de la validación cruzada (accuracy) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nNota: 'p_corregido' es una prueba t remuestreada corregida frente a A. Con pliegues "
          "solapados, p > 0.05 indica que la diferencia no se distingue del ruido.")
    pd.DataFrame(results).to_csv("resultados_cv_por_pliegue.csv", index=False)
    summary.to_csv("resultados_cv_resumen.csv", index=False)

    # --- Ganador entre las redes (los modelos de referencia no compiten) y evaluación en prueba
    nn_names = [NAME_A, NAME_B, NAME_C, NAME_D]
    winner = max(nn_names, key=lambda n: np.mean(results[n]))
    use_early = winner in (NAME_A, NAME_C)
    is_ensemble = winner in (NAME_C, NAME_D)
    print(f"\nGanador por validación cruzada: {winner}")

    cfg_final = cfg_early if use_early else cfg_fixed
    probs, eps = train_nets(X_dev, y_dev, X_test, cfg_final, seeds, n_classes,
                            INNER_VAL_FRAC if use_early else None)
    indiv = [acc(p, y_test) for p in probs]
    ens_acc = acc(np.mean(probs, axis=0), y_test)
    headline = ens_acc if is_ensemble else float(np.mean(indiv))
    n = len(y_test)
    margen = 1.96 * np.sqrt(headline * (1 - headline) / n)

    print("\n=== Evaluación final en PRUEBA (única vez para esta etapa) ===")
    print(f"Redes individuales: {np.mean(indiv):.4f} ± {np.std(indiv):.4f}  (por semilla: "
          + ", ".join(f"{a:.3f}" for a in indiv) + ")")
    print(f"Ensamble de {len(probs)} redes: {ens_acc:.4f}")
    print(f"Resultado del ganador ({winner}): {headline:.4f}  (margen binomial 95% ≈ ±{margen:.3f}, n={n})")
    if prev_test is not None:
        print(f"Resultado anterior (entrenando con 1040 muestras): {prev_test:.4f} -> diferencia {headline - prev_test:+.4f}")
    print(f"UCI: {UCI_BASELINE:.4f} -> diferencia {headline - UCI_BASELINE:+.4f}")

    with open("resultado_final_cv.json", "w") as f:
        json.dump(dict(ganador=winner, acc_test=headline, acc_test_redes=indiv, acc_test_ensamble=ens_acc,
                       epocas_fijas=fixed_epochs, config=cfg_final), f, indent=2)