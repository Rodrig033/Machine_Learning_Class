import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.datasets import load_digits

rng = np.random.default_rng(0)
digits = load_digits()         # 1,797 real 8x8 scans 
X = digits.data / 16           # Pixel brightness 0...1
Y = np.eye(10)[digits.target] 
idx = rng.permutation(len(X))
tr, te = idx[:1400], idx[1400:] # 397 test digits never seen

def layer(n_in, n_out):         # Scaled random init
    return rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_out)), np.zeros(n_out)


W1, b1 = layer(64, 32) # 64 pixel -> 32 neurons
W2, b2 = layer(32, 10) # 32 -> 10 digits


def forward(x):
    h = np.tanh(x @ W1 + b1)
    z = h @ W2 + b2
    p = np.exp(z - z.max(1, keepdims=True)) # What each neuron seens
    return h, p / p.sum(1, keepdims=True)   #  softmax: 10 probabilities

def forward_detallado(x):
    x = np.asarray(x).reshape(1, 64)
    z1 = x @ W1 + b1
    h = np.tanh(z1)
    z2 = h @ W2 + b2
    exp_z = np.exp(z2 - z2.max(axis=1, keepdims=True))
    p = exp_z / exp_z.sum(axis=1, keepdims=True)

    return z1[0], h[0], z2[0], p[0]

def step(lr=0.3):
    global W1, b1, W2, b2
    i = rng.choice(tr, 64)
    x, y = X[i], Y[i]
    h, p = forward(x)
    dz = (p - y) / len(x)
    dh = (dz @ W2.T) * (1 - h ** 2)
    W2 -= lr * h.T @ dz; b2 -= lr * dz.sum(0)
    W1 -= lr * x.T @ dh; b1 -= lr * dh.sum(0)
    return -np.log((p * y).sum(1)).mean()

acc = lambda: (forward(X[te])[1].argmax(1) == digits.target[te]).mean()
history = []
for i in range(300):
    L = step()

    history.append({
        "W1": W1.copy(),
        "loss": L,
         "accuracy": acc()
    })

print("Fotogramas almacenados:", len(history))

def visualizar_entrada(imagen):
    imagen = np.asarray(imagen, dtype=float)

    if imagen.shape != (8, 8):
        raise ValueError("La entrada debe ser una imagen 8x8.")

    if imagen.min() < 0 or imagen.max() > 16:
        raise ValueError("Los píxeles debe de estar entre 0 y 16.")

    x = imagen.reshape(1, 64) / 16 # Normalitation
    z1, h, z2, p = forward_detallado(x)
    prediccion = np.argmax(p)
    confianza = p[prediccion]

    # Figure
    fig, axs = plt.subplots(
        1, 3,
        figsize=(13, 4),
        facecolor="black"
    )

    # 1. Imagen original
    axs[0].imshow(imagen, cmap="gray_r", vmin=0, vmax=16)
    axs[0].set_title("Entrenada: imagen 8x8", color="white")
    axs[0].axis("off")

    # 2. Activaciones de la capa oculta
    axs[1].imshow(
        h.reshape(4, 8),
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    axs[1].set_title("Activaciones: capa oculta", color="white")
    axs[1].set_xticks(range(8))
    axs[1].set_yticks(range(4))

    # 3. Probabilidades de salida
    bars = axs[2].bar(range(10), p)
    axs[2].set_xticks(range(10))
    axs[2].set_ylim(0, 1)
    axs[2].set_title("Probabilidades de clasificación", color="white")
    axs[2].set_xlabel("Digito", color="white")
    axs[2].set_ylabel("Probabilidad", color="white")

    for ax in axs:
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.set_facecolor("#171717")

    # Resaltar la predicción
    bars[prediccion].set_color("limegreen")

    fig.suptitle(
        f"Predicción: {prediccion} Confianza: {confianza:.2%}",
        color="white",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

fig, axs = plt.subplots(4, 8, figsize=(8, 5.4), facecolor="black")
ims = [a.imshow(W1[:, k].reshape(8, 8), cmap="inferno", vmin=1, vmax=1, animated=True)
       for k, a in enumerate(axs.flat)]

for a in axs.flat: a.axis("off")
info = fig.text(.5, .02, "", ha="center", color="w", family="monospace")


def frame(i):
    state = history[i]
    for k, im in enumerate(ims):
        im.set_data(W1[:, k].reshape(8, 8)) # K neuron's 64 heights 
    info.set_text(
            f"step {i + 1:>3} | "
            f"loss: {state['loss']:.3f} | "
            f"test acc: {state['accuracy']:.1%}")

    return ims + [info]

anim = animation.FuncAnimation(fig, frame, frames=len(history), interval=33, blit=False, repeat=False, cache_frame_data=False)

writer = animation.FFMpegWriter(fps=30, codec="libx264", bitrate=2000, extra_args=["-pix_fmt", "yuv420p"])
anim.save("digits.mp4", writer="ffmpeg", fps=30, dpi=120)

# Test de la red neuronal
indice = te[0]
imagen = digits.images[indice]
print("Numero real:", digits.target[indice])
visualizar_entrada(imagen)

plt.close(fig)