import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Cargar el dataset de Iris
iris = load_iris()
X, y = iris.data, iris.target          # 150 muestras, 4 características, clases 0/1/2
print(f"Dataset Iris: {X.shape[0]} muestras, {X.shape[1]} características")

# División 70% entrenamiento / 15% validación / 15% prueba
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print(f"Entrenamiento: {len(X_train)} | Validación: {len(X_val)} | Prueba: {len(X_test)}")

# Escalado: el escalador solo "ve" el conjunto de entrenamiento
scaler = StandardScaler().fit(X_train)
X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))


def to_tensors(X_np, y_np):
    xt = torch.tensor(X_np, dtype=torch.float32)
    yt = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)   # objetivo numérico (n, 1)
    return xt, yt


Xtr, ytr = to_tensors(X_train, y_train)
Xva, yva = to_tensors(X_val, y_val)
Xte, yte = to_tensors(X_test, y_test)

# 4 entradas -> 8 ocultas -> 8 ocultas -> 1 salida
class RedIris(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),     # primera capa oculta
            nn.Linear(8, 8), nn.ReLU(),     # segunda capa oculta
            nn.Linear(8, 1),                # salida: un número (la especie como 0, 1 o 2)
        )

    def forward(self, x):
        return self.net(x)


model = RedIris()
print("\nArquitectura:")
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def to_class(output):
    """Redondea la salida al entero más cercano y la limita a [0, 2]."""
    return output.round().clamp(0, 2).squeeze(1).long()


def accuracy(output, labels_np):
    labels = torch.tensor(labels_np, dtype=torch.long)
    return (to_class(output) == labels).float().mean().item()


# Entrenamiento (con los datos de entrenamiento; la validación se evalúa en cada época pero nunca se usa para actualizar los pesos)
EPOCHS = 300
hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss, best_epoch, best_state = float("inf"), 0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(Xtr), ytr)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out_tr, out_va = model(Xtr), model(Xva)
        hist["train_loss"].append(criterion(out_tr, ytr).item())
        hist["val_loss"].append(criterion(out_va, yva).item())
        hist["train_acc"].append(accuracy(out_tr, y_train))
        hist["val_acc"].append(accuracy(out_va, y_val))

    # Guardar los pesos de la época con menor pérdida de validación
    if hist["val_loss"][-1] < best_val_loss:
        best_val_loss, best_epoch = hist["val_loss"][-1], epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch % 50 == 0 or epoch == 1:
        print(f"Época {epoch:3d} | "
              f"loss train {hist['train_loss'][-1]:.4f}, val {hist['val_loss'][-1]:.4f} | "
              f"acc train {hist['train_acc'][-1]:.3f}, val {hist['val_acc'][-1]:.3f}")

print(f"\nMenor pérdida de validación: {best_val_loss:.4f} (época {best_epoch})")
model.load_state_dict(best_state)

epochs_axis = range(1, EPOCHS + 1)

# Gráfica de la pérdida por época
plt.figure(figsize=(8, 5))
plt.plot(epochs_axis, hist["train_loss"], label="Entrenamiento", linewidth=2)
plt.plot(epochs_axis, hist["val_loss"], label="Validación", linewidth=2)
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Mejor época ({best_epoch})")
plt.xlabel("Época")
plt.ylabel("Pérdida (MSE)")
plt.title("Pérdida por época")
plt.legend()
plt.grid(True)
plt.savefig("iris_perdida.png", dpi=150, bbox_inches="tight")
plt.show()

# Gráfica de convergencia: accuracy por época
plt.figure(figsize=(8, 5))
plt.plot(epochs_axis, hist["train_acc"], label="Entrenamiento", linewidth=2)
plt.plot(epochs_axis, hist["val_acc"], label="Validación", linewidth=2)
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Mejor época ({best_epoch})")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title("Convergencia: accuracy por época")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.savefig("iris_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()

# Evaluación final con el conjunto de prueba (se usa una sola vez)
model.eval()
with torch.no_grad():
    out_te = model(Xte)
    test_loss = criterion(out_te, yte).item()
    test_acc = accuracy(out_te, y_test)
    y_pred = to_class(out_te).numpy()

print("\n" + "=" * 50)
print("EVALUACIÓN EN EL CONJUNTO DE PRUEBA")
print("=" * 50)
print(f"Pérdida (MSE): {test_loss:.4f}")
print(f"Accuracy     : {test_acc:.4f}  ({int(round(test_acc * len(y_test)))}/{len(y_test)} correctas)")
print("\nMatriz de confusión (filas = real, columnas = predicho):")
print(f"clases: {list(iris.target_names)}")
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))