import argparse
import itertools
import json
import os

import pandas as pd

from yeast_data import prepare_data
from yeast_nn import BASE_CFG, evaluate_config, to_tensors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", default=None, help="ruta de yeast.data (opcional)")
    parser.add_argument("--config", default="mejor_configuracion.json")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    data = to_tensors(prepare_data(args.local, seed=42))

    if os.path.exists(args.config):
        with open(args.config) as f:
            best = json.load(f)["config"]
    else:
        best = {**BASE_CFG, "n_layers": 3, "n_neurons": 16, "activation": "tanh",
                "lr": 0.01, "dropout": 0.2}
    print("Configuración base:", json.dumps(best))

    seeds = list(range(args.seeds))
    rows = []
    # BatchNorm sí/no  x  dropout 0 / 0.2  x  lr 0.01 / 0.05 / 0.1
    for bn, dropout, lr in itertools.product([False, True], [0.0, 0.2], [0.01, 0.05, 0.1]):
        cfg = {**best, "batchnorm": bn, "dropout": dropout, "lr": lr}
        res = evaluate_config(data, cfg, seeds)
        rows.append({"batchnorm": bn, "dropout": dropout, "lr": lr, **res})
        print(f"BN={str(bn):5s} dropout={dropout:.1f} lr={lr:<5} -> "
              f"val acc {res['val_acc_media']:.3f} (±{res['val_acc_std']:.3f}), "
              f"train acc {res['train_acc']:.3f}, val loss {res['val_loss_media']:.3f}, "
              f"mejor época {res['mejor_epoca']:.0f}")

    df = pd.DataFrame(rows)
    df.to_csv("resultados_batchnorm.csv", index=False)
    print("\n=== Resumen (ordenado por accuracy de validación) ===")
    print(df.sort_values("val_acc_media", ascending=False)
            .to_string(index=False, float_format=lambda v: f"{v:.4f}"))