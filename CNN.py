import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2  # For image preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Convolutional Layer
class ConvLayer:
    def __init__(self, num_filters, filter_size, stride=1, padding=0, activation_fn=None):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activation_fn = activation_fn
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1

    def forward(self, input_image):
        self.input_image = np.pad(input_image, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        self.output_shape = (
            (self.input_image.shape[0] - self.filter_size) // self.stride + 1,
            (self.input_image.shape[1] - self.filter_size) // self.stride + 1,
        )
        self.output = np.zeros((self.num_filters, *self.output_shape))
        for f in range(self.num_filters):
            for i in range(0, self.output_shape[0]):
                for j in range(0, self.output_shape[1]):
                    region = self.input_image[
                        i * self.stride:i * self.stride + self.filter_size,
                        j * self.stride:j * self.filter_size,
                    ]
                    self.output[f, i, j] = np.sum(region * self.filters[f])
            if self.activation_fn:
                self.output[f] = self.activation_fn(self.output[f])
        return self.output

    def backward(self, d_output):
        # Backward pass for gradients
        pass


# Pooling Layer
class PoolingLayer:
    def __init__(self, pool_size, stride=2, mode='max'):
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

    def forward(self, input_image):
        self.input_image = input_image
        self.output_shape = (
            input_image.shape[0],
            (input_image.shape[1] - self.pool_size) // self.stride + 1,
            (input_image.shape[2] - self.pool_size) // self.stride + 1,
        )
        self.output = np.zeros(self.output_shape)
        for f in range(self.output_shape[0]):
            for i in range(0, self.output_shape[1]):
                for j in range(0, self.output_shape[2]):
                    region = self.input_image[f,
                                              i * self.stride:i * self.stride + self.pool_size,
                                              j * self.stride:j * self.stride + self.pool_size]
                    if self.mode == 'max':
                        self.output[f, i, j] = np.max(region)
                    elif self.mode == 'average':
                        self.output[f, i, j] = np.mean(region)
        return self.output


# Integrating into Network
class CNN:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, d_output):
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)


# Training and Visualization
def train_cnn(cnn, X_train, y_train, epochs, learning_rate):
    for epoch in range(epochs):
        for X, y in zip(X_train, y_train):
            output = cnn.forward(X)
            loss = mean_squared_error(y, output)
            d_output = output - y
            cnn.backward(d_output)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")


# Main Function
def main():
    # Load the dataset
    # Here, load image data and preprocess
    X = []  # Preprocessed image data
    y = []  # Corresponding labels

    # Build the CNN
    cnn = CNN()
    cnn.add_layer(ConvLayer(8, 3, stride=1, padding=1, activation_fn=relu))
    cnn.add_layer(PoolingLayer(2, stride=2, mode='max'))

    # Add fully connected layers or adapt existing code
    # Train and visualize using similar approaches as earlier projects


if __name__ == "__main__":
    main()
