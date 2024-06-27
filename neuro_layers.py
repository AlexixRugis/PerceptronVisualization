import numpy as np

class LinearLayer:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.last_input = np.zeros((n_inputs, 1))
        self.last_output = np.zeros((n_outputs, 1))
        self.weights = np.zeros((n_outputs, n_inputs))
        self.biases = np.zeros((n_outputs, 1))
        
    def randomize_weights(self):
        self.weights = np.random.uniform(-1, 1, (self.n_outputs, self.n_inputs))
        self.biases = np.random.uniform(-1, 1, (self.n_outputs, 1))
        
    def back_propagate(self, der_vector: np.array, learning_rate: float) -> np.array:
        gradients = np.dot(der_vector, self.last_input.T)
        
        derivative = np.dot(self.weights.T, der_vector)
        
        self.biases += der_vector * learning_rate
        self.weights += gradients * learning_rate
        
        return derivative
        
    def __call__(self, inp_vector: np.array) -> np.array:
        self.last_input = inp_vector
        self.last_output = self.weights.dot(inp_vector) + self.biases
        return self.last_output
    
class SigmoidActivator:
    def derivative(self, x: np.array) -> np.array:
        v = self.__call__(x)
        return v * (1.0 - v)
    
    def back_propagate(self, der_vector: np.array, learning_rate: float) -> np.array:
        return der_vector * self.derivative(self.last_input)
    
    def __call__(self, inp_vector: np.array) -> np.array:
        return 1.0 / (1.0 + np.exp(-inp_vector))
    
class TahnActivator:
    def derivative(self, x: np.array) -> np.array:
        v = self.__call__(x)
        return 1.0 - v*v
    
    def back_propagate(self, der_vector: np.array, learning_rate: float) -> np.array:
        return der_vector * self.derivative(self.last_input)
    
    def __call__(self, inp_vector: np.array) -> np.array:
        self.last_input = inp_vector
        return 2.0 / (1.0 + np.exp(-2.0 * inp_vector)) - 1.0
    
class SequentialModel:
    def __init__(self, layers):
        self.layers = layers
    
    def randomize_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'randomize_weights'):
                layer.randomize_weights()
                
    def fit(self, inp_vector: np.array, result: np.array, learning_rate: float):
        predicted = self.__call__(inp_vector)
        
        der = (2.0 / inp_vector.shape[0]) * (result - predicted)
        for layer in self.layers[::-1]:
            der = layer.back_propagate(der, learning_rate)
        
    def __call__(self, inp_vector: np.array) -> np.array:
        for layer in self.layers:
            inp_vector = layer(inp_vector)
            
        return inp_vector