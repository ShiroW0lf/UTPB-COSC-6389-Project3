import os
import numpy as np
import threading
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Loss Function
def categorical_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

def categorical_crossentropy_derivative(y_true, y_pred):
    return y_pred - y_true


# Convolution Layer
class ConvLayer:
    def __init__(self, input_depth, filter_count, filter_size):
        self.filters = np.random.randn(filter_count, input_depth, filter_size, filter_size) * 0.1

    def forward(self, input):
        # Ensure input has a batch dimension
        self.input = input  # (batch_size, input_depth, input_height, input_width)
        batch_size, input_depth, input_height, input_width = input.shape
        filter_count, _, filter_size, _ = self.filters.shape
        output_height = input_height - filter_size + 1
        output_width = input_width - filter_size + 1
        output = np.zeros((batch_size, filter_count, output_height, output_width))

        for b in range(batch_size):
            for f in range(filter_count):
                filter_ = self.filters[f]
                for i in range(output_height):
                    for j in range(output_width):
                        region = input[b, :, i:i + filter_size, j:j + filter_size]
                        output[b, f, i, j] = np.sum(region * filter_)

        self.output = relu(output)
        return self.output

    def backward(self, d_output, learning_rate):
        filter_count, _, filter_size, _ = self.filters.shape
        batch_size, input_depth, input_height, input_width = self.input.shape

        d_filters = np.zeros(self.filters.shape)

        for b in range(batch_size):
            for f in range(filter_count):
                for i in range(input_height - filter_size + 1):
                    for j in range(input_width - filter_size + 1):
                        region = self.input[b, :, i:i + filter_size, j:j + filter_size]
                        d_filters[f] += d_output[b, f, i, j] * region

        self.filters -= learning_rate * d_filters


# Fully Connected Layer
class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        self.output = relu(np.dot(input, self.weights) + self.biases)
        return self.output

    def backward(self, d_output, learning_rate):
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input

def load_images(folder_path, size=(64, 64)):
    images, labels = [], []
    for label, class_name in enumerate(['cats', 'dogs']):
        class_folder = os.path.join(folder_path, class_name)
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            img = Image.open(file_path).convert('L').resize(size)
            images.append(np.array(img) / 255.0)
            labels.append(label)
    images = np.expand_dims(np.array(images), axis=1)
    labels = np.eye(2)[labels]  # One-hot encode
    return images, labels

train_images, train_labels = load_images('data/train')
test_images, test_labels = load_images('data/test')

class CNN:
    def __init__(self):
        self.conv1 = ConvLayer(1, 8, 3)  # 8 filters, 3x3 size
        self.fc1 = FullyConnectedLayer(8 * 62 * 62, 128)  # 128 neurons
        self.fc2 = FullyConnectedLayer(128, 2)  # 2 output classes

    def forward(self, input):
        conv_output = self.conv1.forward(input)
        flattened = conv_output.flatten().reshape(1, -1)
        fc1_output = self.fc1.forward(flattened)
        fc2_output = softmax(self.fc2.forward(fc1_output))
        return fc2_output

    def backward(self, input, y_true, y_pred, learning_rate):
        d_output = categorical_crossentropy_derivative(y_true, y_pred)
        d_fc2 = self.fc2.backward(d_output, learning_rate)
        d_fc1 = self.fc1.backward(d_fc2, learning_rate)
        d_conv = d_fc1.reshape(self.conv1.output.shape)
        self.conv1.backward(d_conv, learning_rate)

    def train(self, images, labels, epochs, learning_rate):
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(images)):
                input = images[i:i+1]
                y_true = labels[i:i+1]
                y_pred = self.forward(input)
                epoch_loss += categorical_crossentropy(y_true, y_pred)
                self.backward(input, y_true, y_pred, learning_rate)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(images)}")

    def evaluate(self, images, labels):
        correct = 0
        for i in range(len(images)):
            y_true = np.argmax(labels[i])
            y_pred = np.argmax(self.forward(images[i:i+1]))
            correct += int(y_true == y_pred)
        return correct / len(images)

# Train and evaluate the model
cnn = CNN()
cnn.train(train_images, train_labels, epochs=10, learning_rate=0.01)
accuracy = cnn.evaluate(test_images, test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


class CNNWithVisualization(CNN):
    def __init__(self, gui_update_callback):
        super().__init__()
        self.gui_update_callback = gui_update_callback  # Callback to update GUI

    def train_with_visualization(self, images, labels, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(images)):
                input = images[i:i + 1]
                y_true = labels[i:i + 1]
                y_pred = self.forward(input)
                epoch_loss += categorical_crossentropy(y_true, y_pred)
                self.backward(input, y_true, y_pred, learning_rate)
            average_loss = epoch_loss / len(images)
            losses.append(average_loss)

            # Evaluate model and send updates to GUI
            accuracy = self.evaluate(images, labels)
            self.gui_update_callback(epoch + 1, average_loss, accuracy, losses)

    def evaluate(self, images, labels):
        correct = 0
        for i in range(len(images)):
            y_true = np.argmax(labels[i])
            y_pred = np.argmax(self.forward(images[i:i + 1]))
            correct += int(y_true == y_pred)
        return correct / len(images)

class TrainingGUI:
    def __init__(self, cnn_model):
        self.root = tk.Tk()
        self.root.title("CNN Training Visualization")
        self.cnn_model = cnn_model

        # Layout Frames
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Matplotlib Plot
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        self.line, = self.ax.plot([], [], label="Loss")
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack()

        # Training Info
        self.epoch_label = ttk.Label(self.info_frame, text="Epoch: 0", font=("Arial", 14))
        self.epoch_label.pack(pady=5)

        self.loss_label = ttk.Label(self.info_frame, text="Loss: 0.00", font=("Arial", 14))
        self.loss_label.pack(pady=5)

        self.accuracy_label = ttk.Label(self.info_frame, text="Accuracy: 0.00%", font=("Arial", 14))
        self.accuracy_label.pack(pady=5)

        # CNN Architecture Visualization
        self.architecture_label = ttk.Label(self.info_frame, text="CNN Architecture", font=("Arial", 16, "bold"))
        self.architecture_label.pack(pady=10)

        self.architecture_text = tk.Text(self.info_frame, width=40, height=10, font=("Courier", 10))
        self.architecture_text.pack()
        self.architecture_text.insert(
            "1.0",
            "Conv Layer: 8 filters, 3x3\n"
            "Flatten\n"
            "Fully Connected: 128 neurons\n"
            "Fully Connected: 2 neurons (softmax)"
        )
        self.architecture_text.configure(state="disabled")

    def update_plot(self, epoch, loss_history):
        self.line.set_data(range(1, epoch + 1), loss_history)
        self.ax.set_xlim(1, epoch)
        self.ax.set_ylim(0, max(loss_history) * 1.1)
        self.canvas.draw()

    def update_info(self, epoch, loss, accuracy):
        self.epoch_label.config(text=f"Epoch: {epoch}")
        self.loss_label.config(text=f"Loss: {loss:.4f}")
        self.accuracy_label.config(text=f"Accuracy: {accuracy * 100:.2f}%")

    def gui_update_callback(epoch, loss, accuracy, losses):
        gui.update_plot(epoch, losses)
        gui.update_info(epoch, loss, accuracy)

    # Instantiate CNN and GUI
    cnn_model = CNNWithVisualization(gui_update_callback)
    gui = TrainingGUI(cnn_model)


    def train_model():
        train_images, train_labels = load_images('data/train')
        cnn_model.train_with_visualization(train_images, train_labels, epochs=10, learning_rate=0.01)

    training_thread = threading.Thread(target=train_model)
    training_thread.start()

    # Start the GUI
    gui.run()

    def run(self):
        self.root.mainloop()

