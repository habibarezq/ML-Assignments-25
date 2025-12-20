import numpy as np

class Autoencoder:
    def __init__(self, input_dim, hidden_dims, bottleneck_dim, activation='relu', learning_rate=0.01, l2_lambda=0.001, lr_decay_rate=0.9, lr_decay_steps=10):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

        encoder_dims = [input_dim] + hidden_dims + [bottleneck_dim]
        self.encoder_weights = []
        self.encoder_biases = []
        for i in range(len(encoder_dims) - 1):
            self.encoder_weights.append(np.random.randn(encoder_dims[i], encoder_dims[i+1]) * 0.01)
            self.encoder_biases.append(np.zeros((1, encoder_dims[i+1])))

        decoder_dims = [bottleneck_dim] + hidden_dims[::-1] + [input_dim]
        self.decoder_weights = []
        self.decoder_biases = []
        for i in range(len(decoder_dims) - 1):
            self.decoder_weights.append(np.random.randn(decoder_dims[i], decoder_dims[i+1]) * 0.01)
            self.decoder_biases.append(np.zeros((1, decoder_dims[i+1])))

        self.activation_func = {
            'relu': lambda x: np.maximum(0, x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x)
        }[activation]

        self.activation_deriv = {
            'relu': lambda x: (x > 0).astype(float),
            'sigmoid': lambda x: x * (1 - x),
            'tanh': lambda x: 1 - x**2
        }[activation]

    def forward(self, X):
        encoder_activations = [X]
        encoder_zs = []
        for w, b in zip(self.encoder_weights, self.encoder_biases):
            z = encoder_activations[-1] @ w + b
            encoder_zs.append(z)
            encoder_activations.append(self.activation_func(z))

        decoder_activations = [encoder_activations[-1]]
        decoder_zs = []
        for w, b in zip(self.decoder_weights, self.decoder_biases):
            z = decoder_activations[-1] @ w + b
            decoder_zs.append(z)
            decoder_activations.append(self.activation_func(z))

        return decoder_activations[-1], encoder_activations, encoder_zs, decoder_activations, decoder_zs

    def backward(self, X, output, encoder_activations, encoder_zs, decoder_activations, decoder_zs):
        batch_size = X.shape[0]
        dL_da = (output - X) / batch_size

        decoder_dweights = []
        decoder_dbiases = []
        for i in reversed(range(len(self.decoder_weights))):
            dL_dz = dL_da * self.activation_deriv(decoder_zs[i])
            decoder_dweights.insert(0, decoder_activations[i].T @ dL_dz + self.l2_lambda * self.decoder_weights[i])
            decoder_dbiases.insert(0, np.sum(dL_dz, axis=0, keepdims=True))
            dL_da = dL_dz @ self.decoder_weights[i].T

        encoder_dweights = []
        encoder_dbiases = []
        for i in reversed(range(len(self.encoder_weights))):
            dL_dz = dL_da * self.activation_deriv(encoder_zs[i])
            encoder_dweights.insert(0, encoder_activations[i].T @ dL_dz + self.l2_lambda * self.encoder_weights[i])
            encoder_dbiases.insert(0, np.sum(dL_dz, axis=0, keepdims=True))
            dL_da = dL_dz @ self.encoder_weights[i].T

        return encoder_dweights, encoder_dbiases, decoder_dweights, decoder_dbiases

    def update_weights(self, enc_dw, enc_db, dec_dw, dec_db, epoch):
        lr = self.learning_rate * (self.lr_decay_rate ** (epoch // self.lr_decay_steps))
        for i in range(len(self.encoder_weights)):
            self.encoder_weights[i] -= lr * enc_dw[i]
            self.encoder_biases[i] -= lr * enc_db[i]
        for i in range(len(self.decoder_weights)):
            self.decoder_weights[i] -= lr * dec_dw[i]
            self.decoder_biases[i] -= lr * dec_db[i]

    def train(self, X, epochs=100, batch_size=32):
        n = X.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X = X[perm]
            epoch_loss = 0
            for i in range(0, n, batch_size):
                Xb = X[i:i+batch_size]
                out, ea, ez, da, dz = self.forward(Xb)
                mse = np.mean((out - Xb) ** 2)
                l2 = self.l2_lambda * sum(np.sum(w ** 2) for w in self.encoder_weights + self.decoder_weights)
                epoch_loss += mse + l2
                enc_dw, enc_db, dec_dw, dec_db = self.backward(Xb, out, ea, ez, da, dz)
                self.update_weights(enc_dw, enc_db, dec_dw, dec_db, epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (n // batch_size):.4f}")

    def encode(self, X):
        a = X
        for w, b in zip(self.encoder_weights, self.encoder_biases):
            a = self.activation_func(a @ w + b)
        return a

    def decode(self, Z):
        a = Z
        for w, b in zip(self.decoder_weights, self.decoder_biases):
            a = self.activation_func(a @ w + b)
        return a


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(1000, 784)
    ae = Autoencoder(784, [256, 128, 64], 32, activation='relu')
    ae.train(X, epochs=50, batch_size=64)
    z = ae.encode(X[:5])
    x_hat = ae.decode(z)
    print(z.shape, x_hat.shape)
