import argparse
import copy
import itertools
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from yeast_data import prepare_data

torch.set_num_threads(1)      # para redes tan pequeñas suele ser más rápido

UCI_BASELINE = 0.5687

# Configuración base; la búsqueda y los barridos sustituyen algunos de estos valores
BASE_CFG = dict(
    n_layers=2,          # capas ocultas
    n_neurons=32,        # neuronas por capa oculta
    activation="relu",
    lr=0.01,
    weight_decay=0.0,    # regularización L2
    dropout=0.0,
    epochs=300,          # máximo de épocas
    batch_size=32,
    patience=30,         # parada temprana: épocas sin mejorar la pérdida de validación (None = sin parada)
    restore_best=True,   # al final, restaurar los pesos de la mejor época de validación
)

FULL_GRID = dict(
    n_layers=[1, 2, 3],
    n_neurons=[16, 32, 64],
    activation=["relu", "tanh"],
    lr=[0.001, 0.01],
    dropout=[0.0, 0.2],
)
QUICK_GRID = dict(n_layers=[1, 2], n_neurons=[16, 32], activation=["relu"], lr=[0.01])

SWEEPS = {
    "lr": [0.0001, 0.001, 0.01, 0.05, 0.1],
    "epochs": [50, 100, 300, 1000],              # sin parada temprana ni restauración
    "batch_size": [16, 32, 64, 128],
    "weight_decay": [0.0, 1e-4, 1e-3, 1e-2],
    "dropout": [0.0, 0.1, 0.2, 0.4],
    "activation": ["relu", "tanh", "sigmoid", "leaky_relu", "elu"],
}
QUICK_SWEEPS = {"lr": [0.001, 0.01], "epochs": [50, 100]}

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
               "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}


# Modelo y entrenamiento
def build_model(n_in, n_out, n_layers, n_neurons, activation, dropout):
    """MLP: n_in -> [Linear -> activación -> (Dropout)] x n_layers -> Linear(n_out).
    La última capa no lleva activación: CrossEntropyLoss ya aplica la softmax."""
    layers, d = [], n_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, n_neurons), ACTIVATIONS[activation]()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = n_neurons
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    out = model(X)
    loss = nn.functional.cross_entropy(out, y).item()
    pred = out.argmax(dim=1)
    return loss, (pred == y).float().mean().item(), pred.numpy()


def train_model(data, cfg, seed):
    torch.manual_seed(seed)
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]

    model = build_model(Xtr.shape[1], data["n_classes"], cfg["n_layers"], cfg["n_neurons"],
                        cfg["activation"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    n = len(Xtr)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, cfg["batch_size"]):
            idx = perm[s:s + cfg["batch_size"]]
            opt.zero_grad()
            criterion(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()

        tr_l, tr_a, _ = evaluate(model, Xtr, ytr)
        va_l, va_a, _ = evaluate(model, Xva, yva)
        hist["train_loss"].append(tr_l); hist["val_loss"].append(va_l)
        hist["train_acc"].append(tr_a); hist["val_acc"].append(va_a)

        if va_l < best_loss - 1e-4:
            best_loss, best_epoch, wait = va_l, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if cfg["patience"] and wait >= cfg["patience"]:
                break

    if cfg["restore_best"] and best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_epoch


def evaluate_config(data, cfg, seeds):
    """Entrena la configuración con varias semillas; devuelve métricas de train y validación."""
    t0 = time.time()
    tr_acc, va_acc, va_loss, best_ep = [], [], [], []
    for s in seeds:
        model, hist, b = train_model(data, cfg, s)
        _, a_tr, _ = evaluate(model, *data["train"])
        l_va, a_va, _ = evaluate(model, *data["val"])
        tr_acc.append(a_tr); va_acc.append(a_va); va_loss.append(l_va); best_ep.append(b)
    return dict(train_acc=np.mean(tr_acc), val_acc_media=np.mean(va_acc), val_acc_std=np.std(va_acc),
                val_loss_media=np.mean(va_loss), mejor_epoca=np.mean(best_ep),
                segundos=time.time() - t0)


# Etapa 1: búsqueda en rejilla
def grid_search(data, grid, seeds):
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    results = []
    print(f"\n=== Búsqueda en rejilla: {len(combos)} configuraciones x {len(seeds)} semillas ===")
    for i, values in enumerate(combos, 1):
        cfg = {**BASE_CFG, **dict(zip(keys, values))}
        res = evaluate_config(data, cfg, seeds)
        results.append((cfg, res))
        desc = ", ".join(f"{k}={cfg[k]}" for k in keys)
        print(f"[{i:3d}/{len(combos)}] {desc} -> val acc {res['val_acc_media']:.3f} "
              f"(±{res['val_acc_std']:.3f}), train acc {res['train_acc']:.3f}, {res['segundos']:.0f}s")
        rows = [{**{k: c[k] for k in keys}, **r} for c, r in results]
        pd.DataFrame(rows).to_csv("resultados_grid.csv", index=False)   # se guarda en cada paso

    best_cfg, best_res = max(results, key=lambda t: (t[1]["val_acc_media"], -t[1]["val_loss_media"]))
    df = pd.DataFrame([{**{k: c[k] for k in keys}, **r} for c, r in results])
    df = df.sort_values(["val_acc_media", "val_loss_media"], ascending=[False, True])
    print("\nMejores 10 configuraciones (por accuracy de validación):")
    print(df.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return best_cfg


# Etapa 2: sensibilidad a un hiperparámetro a la vez
def sensitivity(data, best_cfg, sweeps, seeds):
    rows = []
    print("\n=== Sensibilidad (un hiperparámetro a la vez, el resto = mejor configuración) ===")
    for param, values in sweeps.items():
        for v in values:
            cfg = {**best_cfg, param: v}
            if param == "epochs":
                # para ver el efecto de las épocas: sin parada temprana y sin restaurar la mejor época
                cfg["patience"], cfg["restore_best"] = None, False
            res = evaluate_config(data, cfg, seeds)
            rows.append(dict(hiperparametro=param, valor=v, **res))
            print(f"{param:13s} = {str(v):10s} -> val acc {res['val_acc_media']:.3f} "
                  f"(±{res['val_acc_std']:.3f}), train acc {res['train_acc']:.3f}, "
                  f"val loss {res['val_loss_media']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv("resultados_sensibilidad.csv", index=False)
    return df


# Etapa 3: evaluación final en prueba
def final_evaluation(data, cfg, seeds):
    print("\n=== Evaluación final en el conjunto de PRUEBA ===")
    print("Configuración elegida:", json.dumps(cfg))
    accs, first = [], None
    for i, s in enumerate(seeds):
        model, hist, best_ep = train_model(data, cfg, s)
        loss, acc, pred = evaluate(model, *data["test"])
        accs.append(acc)
        print(f"  semilla {s}: acc test = {acc:.4f}, pérdida = {loss:.4f}, mejor época = {best_ep}")
        if i == 0:
            first = (hist, best_ep, pred)

    n = len(data["test"][1])
    mean, std = float(np.mean(accs)), float(np.std(accs))
    margen = 1.96 * np.sqrt(mean * (1 - mean) / n)
    print(f"\nAccuracy en prueba: {mean:.4f} ± {std:.4f} (media ± desv. entre {len(seeds)} semillas)")
    print(f"Margen de error binomial aproximado al 95% con {n} muestras: ±{margen:.3f}")
    print(f"UCI reporta {UCI_BASELINE:.4f} -> diferencia: {mean - UCI_BASELINE:+.4f}")

    hist, best_ep, pred = first
    y_true = data["test"][1].numpy()
    classes = list(data["classes"])
    print(f"\nReporte por clase (semilla {seeds[0]}):")
    print(classification_report(y_true, pred, labels=range(len(classes)),
                                target_names=classes, zero_division=0))
    plot_results(hist, best_ep, y_true, pred, classes)
    with open("mejor_configuracion.json", "w") as f:
        json.dump(dict(config=cfg, acc_test_media=mean, acc_test_std=std, acc_test_por_semilla=accs), f, indent=2)


def plot_results(hist, best_ep, y_true, pred, classes):
    epochs = range(1, len(hist["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist["train_loss"], label="Entrenamiento", linewidth=2)
    plt.plot(epochs, hist["val_loss"], label="Validación", linewidth=2)
    plt.axvline(best_ep, color="gray", linestyle="--", label=f"Mejor época ({best_ep})")
    plt.xlabel("Época"); plt.ylabel("Pérdida (entropía cruzada)")
    plt.title("Pérdida por época"); plt.legend(); plt.grid(True)
    plt.savefig("yeast_perdida.png", dpi=150, bbox_inches="tight"); plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist["train_acc"], label="Entrenamiento", linewidth=2)
    plt.plot(epochs, hist["val_acc"], label="Validación", linewidth=2)
    plt.axhline(UCI_BASELINE, color="red", linestyle=":", label=f"UCI ({UCI_BASELINE:.4f})")
    plt.axvline(best_ep, color="gray", linestyle="--", label=f"Mejor época ({best_ep})")
    plt.xlabel("Época"); plt.ylabel("Accuracy")
    plt.title("Convergencia: accuracy por época"); plt.legend(); plt.grid(True)
    plt.savefig("yeast_accuracy.png", dpi=150, bbox_inches="tight"); plt.show()

    cm = confusion_matrix(y_true, pred, labels=range(len(classes)))
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xlabel("Clase predicha"); plt.ylabel("Clase real")
    plt.title("Matriz de confusión (prueba)")
    plt.savefig("yeast_confusion.png", dpi=150, bbox_inches="tight"); plt.show()


def to_tensors(d):
    def t(X, y):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    return dict(train=t(d["X_train"], d["y_train"]), val=t(d["X_val"], d["y_val"]),
                test=t(d["X_test"], d["y_test"]),
                classes=d["classes"], n_classes=len(d["classes"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", default=None, help="ruta de yeast.data (opcional)")
    parser.add_argument("--quick", action="store_true", help="búsqueda reducida para probar")
    parser.add_argument("--seeds", type=int, default=3, help="semillas por configuración en la búsqueda")
    parser.add_argument("--final-seeds", type=int, default=5, help="semillas en la evaluación final")
    args = parser.parse_args()

    raw = prepare_data(args.local, seed=42)       # la partición es fija; cambian las semillas del modelo
    data = to_tensors(raw)
    print(f"Entrenamiento: {len(data['train'][1])} | Validación: {len(data['val'][1])} | "
          f"Prueba: {len(data['test'][1])} | Clases: {list(data['classes'])}")

    search_seeds = list(range(args.seeds))
    best_cfg = grid_search(data, QUICK_GRID if args.quick else FULL_GRID, search_seeds)
    sensitivity(data, best_cfg, QUICK_SWEEPS if args.quick else SWEEPS, search_seeds)
    final_evaluation(data, best_cfg, list(range(args.final_seeds)))