from jax.experimental.mosaic.gpu.fragmented_array import WGMMA_LAYOUT_UPCAST_2X
from sklearn.datasets import fetch_openml
import numpy as np
from jax import grad

mnist = fetch_openml('mnist_784', version=1)


def relu(x):
    return np.maximum(0, x)

def optim(W_1, W_2, loss):
    W_1 -= loss * 0.00001
    W_2 -= loss * 0.00001
    return W_1, W_2
def main (cfg):
    X = np.array(mnist.data.astype("float32"))
    y = np.array(np.eye(10)[mnist.target.astype("int64")])
    print(X.shape, y.shape)

    # init params
    W_1 = np.random.rand(X.shape[1], cfg.hidden)
    W_2 = np.random.rand(cfg.hidden, 10)
    y_hat = relu(X @ W_1) @ W_2
    print(loss_fn(y, y_hat))

    for epoch in range(cfg.epochs):
        y_hat = relu(X @ W_1) @ W_2
        loss = loss_fn(y, y_hat)
        W_1, W_2 = optim(W_1, W_2, loss)
        print(loss)


def loss_fn(y, y_hat):
    return np.abs(y - y_hat).mean()
