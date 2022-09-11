import random # Esta librería permite generar números pseudoaleatorios
import numpy as np # Esta librería contiene una colección de funciones matemáticas y permite crear vectores y matrices multidimensionales

class Network(object): # La clase nos permite construir diferentes objetos (ejemplo del carro) utilizando una vez las instrucciones, para ello se definen variables

    def __init__(self, sizes): # Sizes contiene el número de neuronas en las respectivas capas de la red, además, los parámetros de la función de costo (w,b) son iniciados de manera aleatoria.
        
        self.num_layers = len(sizes) #Número de capas
        self.sizes = sizes 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a): # Se ingresa un dato y encuentra la salida correspondiente
        for b, w in zip(self.biases, self.weights): 
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, #Parámetros de la función SGD
            test_data=None):
        training_data = list(training_data) # Lista de tuplas (x,y) que representa las entradas y salidas deseadas
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta): # Actualiza los pesos y los biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # Calcula las derivadas parciales
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y): # Algoritmo BackPropagation, nos da el gradiente de la función de costo
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # Almacena todas las activaciones capa por capa
        zs = [] # Almacena los vectores z capa por capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data): # Nos dice los resultados que fueron correctos a partir de los datos de prueba
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y): # Derivada de la función de costo
        return (output_activations-y)

# Funciones Auxiliares 
def sigmoid(z): # Función sigmoide
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z): # Derivada de la función sigmoide
    return sigmoid(z)*(1-sigmoid(z))
