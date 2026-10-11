import numpy as np
import matplotlib.pyplot as plt

class ActivationFunctions:
    """Clase que contiene todas las funciones de activación"""

    @staticmethod
    def step_function(x, threshold=0): #Función de paso
      return np.where(x >= threshold, 1, 0)

    @staticmethod
    def step_function_bipolar(x, threshold=0):
      """
      Función escalón bipolar (-1, 1)
      """
      return np.where(x >= threshold, 1, -1)    

    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, ActivationFunctions.elu(x, alpha) + alpha)
    
    @staticmethod
    def swish(x, beta=1.0):
        return x * ActivationFunctions.sigmoid(beta * x)
    
    @staticmethod
    def swish_derivative(x, beta=1.0):
        sig = ActivationFunctions.sigmoid(beta * x)
        return sig + beta * x * sig * (1 - sig)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0)
    
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def mish(x):
        return x * np.tanh(np.log(1 + np.exp(x)))

def plot_activation_functions():
    """Función para visualizar todas las funciones de activación"""
    x = np.linspace(-3, 3, 1000)
    functions = {
        'Linear': ActivationFunctions.linear(x),
        'Sigmoid': ActivationFunctions.sigmoid(x),
        'Tanh': ActivationFunctions.tanh(x),
        'ReLU': ActivationFunctions.relu(x),
        'SoftMax': ActivationFunctions.softmax(x),
        'Leaky ReLU': ActivationFunctions.leaky_relu(x),
        'ELU': ActivationFunctions.elu(x),
        'Swish': ActivationFunctions.swish(x),
        'GELU': ActivationFunctions.gelu(x),
        'Mish': ActivationFunctions.mish(x),
        'Step Function Bipolar' : ActivationFunctions.step_function_bipolar(x)
    }
    
    fig, axes = plt.subplots(3, 4, figsize=(13, 7))
    axes = axes.ravel()
    # bucle para mostrar graficamente todas las funciones 
    for i, (name, y) in enumerate(functions.items()):
        axes[i].plot(x, y, 'b-', linewidth=2, label=name)
        axes[i].set_title(name)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(-3, 3)
        axes[i].axhline(0, color='black', linewidth=0.5)
        axes[i].axvline(0, color='black', linewidth=0.5)
        axes[i].legend()

    # Ocultar el espacio sobrante
    for i in range(len(functions), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Visualizar funciones
    plot_activation_functions()
    
    # Probar funciones individualmente
    x_test = np.array([-2, -1, 0, 1, 2])   
    print("Prueba de funciones de activación:")
    print(f"Input: {x_test}")
    print(f"Linear: {ActivationFunctions.linear(x_test)}")
    print(f"ReLU: {ActivationFunctions.relu(x_test)}")
    print(f"Sigmoid: {ActivationFunctions.sigmoid(x_test)}")
    print(f"Tanh: {ActivationFunctions.tanh(x_test)}")
    print(f"Softmax: {ActivationFunctions.softmax(x_test)}")    
    #Complementar con las funciones faltantes
    print(f"Leaky ReLU: {ActivationFunctions.leaky_relu(x_test)}")
    print(f"Swish: {ActivationFunctions.swish(x_test)}")
    print(f"ELU: {ActivationFunctions.elu(x_test)}")
    print(f"GELU: {ActivationFunctions.gelu(x_test)}")
    print(f"Mush: {ActivationFunctions.mish(x_test)}")
    print(f"Step Function Bipolar: {ActivationFunctions.step_function_bipolar(x_test)}")
