import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo

COLS = ["Sequence_Name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "localization_site"]

FEATURES = COLS[1:-1]

def load_yeast(local_file=None):
    if local_file:
        df = pd.read_csv(local_file, sep=r"\s+", header=None, name=COLS)
        X, y = df[FEATURES], df["localization_site"]
    else: 
        yeast = fetch_ucirepo(id=110)
        X = yeast.data.features
        y = yeast.data.targets.iloc[:, 0]
    X = X.drop(columns=["Sequence_Name"], errors="ignore").astype(float)
    y = y.astype(str).str.strip()
    return X.reset_index(drop=True), y.reset_index(drop=True)


def stratified_split(y, seed=42, val_frac=0.15, test_frac=0.15):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n = len(idx)
        assert n >= 3, f"La clase {c} tiene solo {n} muestras"
        n_val = max(1, int(round(n * val_frac)))
        n_te = max(1, int(round(n * test_frac)))
        va.extend(idx[:n_val])
        te.extend(idx[n_val:n_val + n_te])
        tr.extend(idx[n_val + n_te:])
    rng.shuffle(tr), rng.shuffle(va), rng.shuffle(te)
    return np.array(tr), np.array(va), np.array(te)

def prepare_data(local_file=None, seed=42):
    X, y = load_yeast(local_file)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    i_tr, i_va, i_te = stratified_split(y_enc, seed)
    X_np = X.values

    scaler = StandardScaler().fit(X_np[i_tr])

    return dict(
        X_train=scaler.transform(X_np[i_tr]), y_train=y_enc[i_tr],
        X_val=scaler.transform(X_np[i_va]), y_val=y_enc[i_va],
        X_test=scaler.transform(X_np[i_te]), y_test=y_enc[i_te],
        classes=le.classes_, features=list(X.columns), X_raw=X, y_raw=y,
    )


if __name__ == "__name__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", default=None, help="ruta de yeast.data (opcional)")
    args = parser.parse_args()

    d = prepare_data(args.local)
    X, y = d["X_raw"], d["y_raw"]

    print(f"Muestras: {len(X)} | Características: {X.shape[1]} | Clases: {len(d['classes'])}")
    print(f"Características: {d['features']}")
    print("\nResumen estadístico:")
    print(X.describe().round(3).to_string())
 
    counts = y.value_counts()
    print("\nDistribución de clases:")
    print(pd.DataFrame({"muestras": counts, "%": (counts / len(y) * 100).round(1)}).to_string())
    print(f"\nLínea base (predecir siempre '{counts.index[0]}'): {counts.iloc[0] / len(y):.3f}")
 
    print("\nTamaño de cada conjunto por clase:")
    table = pd.DataFrame({
        "train": pd.Series(d["y_train"]).value_counts(),
        "val": pd.Series(d["y_val"]).value_counts(),
        "test": pd.Series(d["y_test"]).value_counts(),
    }).fillna(0).astype(int).sort_index()
    table.index = d["classes"][table.index]
    table.loc["TOTAL"] = table.sum()
    print(table.to_string())