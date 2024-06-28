import numpy as np

class LinearLayer:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.last_input = np.zeros((n_inputs, 1))
        self.last_output = np.zeros((n_outputs, 1))
        self.weights = np.zeros((n_outputs, n_inputs))
        self.biases = np.zeros((n_outputs, 1))
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)
        self.t = 0

    def randomize_weights(self):
        limit = np.sqrt(6 / (self.n_inputs + self.n_outputs))  
        self.weights = np.random.uniform(-limit, limit, (self.n_outputs, self.n_inputs))
        self.biases = np.zeros((self.n_outputs, 1))

    def back_propagate(self, der_vector: np.array, learning_rate: float, beta1=0.9, beta2=0.999, epsilon=1e-8) -> np.array:
        gradients = np.dot(der_vector, self.last_input.T)
        derivative = np.dot(self.weights.T, der_vector)

        self.t += 1
        self.m_w = beta1 * self.m_w + (1 - beta1) * gradients
        self.v_w = beta2 * self.v_w + (1 - beta2) * gradients ** 2
        m_w_hat = self.m_w / (1 - beta1 ** self.t)
        v_w_hat = self.v_w / (1 - beta2 ** self.t)

        self.m_b = beta1 * self.m_b + (1 - beta1) * der_vector
        self.v_b = beta2 * self.v_b + (1 - beta2) * der_vector ** 2
        m_b_hat = self.m_b / (1 - beta1 ** self.t)
        v_b_hat = self.v_b / (1 - beta2 ** self.t)

        self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

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
    
class TanhActivator:
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
        self.history = {'loss': [], 'accuracy': []}
    
    def randomize_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'randomize_weights'):
                layer.randomize_weights()
                
    def fit(self, inp_vector: np.array, result: np.array, learning_rate: float, epochs: int=1):
        for epoch in range(epochs):
            predicted = self.__call__(inp_vector)
            loss = np.mean((result - predicted) ** 2)
            accuracy = np.mean(np.argmax(predicted, axis=0) == np.argmax(result, axis=0))
            self.history['loss'].append(loss)
            self.history['accuracy'].append(accuracy)
            der = (2.0 / inp_vector.shape[0]) * (predicted - result)
            for layer in self.layers[::-1]:
                der = layer.back_propagate(der, learning_rate)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
    def __call__(self, inp_vector: np.array) -> np.array:
        for layer in self.layers:
            inp_vector = layer(inp_vector)
        return inp_vector
    
    def evaluate(self, inp_vector: np.array, result: np.array) -> float:
        predicted = self.__call__(inp_vector)
        loss = np.mean((result - predicted) ** 2)
        accuracy = np.mean(np.argmax(predicted, axis=0) == np.argmax(result, axis=0))
        return loss, accuracy
