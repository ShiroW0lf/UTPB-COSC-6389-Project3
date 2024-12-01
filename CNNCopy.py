import os
import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image


# Custom CNN Implementation
class CNNWithVisualization:
    def __init__(self, gui_update_callback):
        self.gui_update_callback = gui_update_callback
        self.weights = {}
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights for the CNN layers."""
        print("Initializing weights...")
        self.weights['conv'] = np.random.randn(3, 3, 3, 8)  # Conv layer weights
        print(f"Conv layer weights initialized with shape: {self.weights['conv'].shape}")

        # Calculate the actual size of the flattened input after the conv layer
        conv_output_height = 64 - 3 + 1  # Assuming input height = 64, filter size = 3, stride = 1
        conv_output_width = 64 - 3 + 1  # Assuming input width = 64, filter size = 3, stride = 1
        flattened_size = conv_output_height * conv_output_width * 8  # 8 filters
        print(f"Flattened size after conv layer: {flattened_size}")

        # Fully connected layer weights
        output_neurons = 2  # Binary classification
        self.weights['fc'] = np.random.randn(flattened_size, output_neurons) * np.sqrt(2 / flattened_size)
        print(f"Fully connected weights initialized with shape: {self.weights['fc'].shape}")

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, inputs):
        """Perform forward propagation."""
        print("Performing forward propagation...")
        conv_output = self.convolve(inputs, self.weights['conv'])
        print(f"Conv layer output shape: {conv_output.shape}")
        flattened = conv_output.reshape(conv_output.shape[0], -1)
        print(f"Flattened output shape: {flattened.shape}")
        fc_output = flattened.dot(self.weights['fc'])
        print(f"Fully connected output shape: {fc_output.shape}")
        return self.softmax(fc_output)
        relu = lambda x: np.maximum(0, x)  # ReLU activation
        conv_activated = relu(conv_output)

    def convolve(self, inputs, filters):
        """Simulate convolution."""
        batch_size, height, width, channels = inputs.shape
        filter_height, filter_width, _, num_filters = filters.shape
        output_height = height - filter_height + 1
        output_width = width - filter_width + 1
        conv_output = np.zeros((batch_size, output_height, output_width, num_filters))
        for i in range(output_height):
            for j in range(output_width):
                region = inputs[:, i:i+filter_height, j:j+filter_width, :]
                conv_output[:, i, j, :] = np.tensordot(region, filters, axes=([1, 2, 3], [0, 1, 2]))
        return self.relu(conv_output)

    def compute_loss(self, predictions, labels):
        """Compute categorical cross-entropy loss."""
        m = labels.shape[0]
        return -np.sum(labels * np.log(predictions + 1e-8)) / m

    def compute_accuracy(self, predictions, labels):
        """Compute accuracy."""
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

    def train_with_visualization(self, train_images, train_labels, epochs=10, learning_rate=0.01):
        """Train the CNN."""
        losses = []
        for epoch in range(1, epochs + 1):
            predictions = self.forward_propagation(train_images)
            loss = self.compute_loss(predictions, train_labels)
            losses.append(loss)
            accuracy = self.compute_accuracy(predictions, train_labels)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy * 100:.2f}%")
            self.gui_update_callback(epoch, loss, accuracy, losses)

            # Simple gradient update (placeholder for backpropagation)
            # This is where the backpropagation would be implemented

    @staticmethod
    def predict(images):
        """Simulate predictions."""
        return np.random.randint(0, 2, size=len(images))


# Load dataset
def load_images(directory, image_size=(64, 64)):
    print(f"Loading images from {directory}...")
    images, labels = [], []
    for label, subfolder in enumerate(['cats', 'dogs']):
        subfolder_path = os.path.join(directory, subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.jpg'):
                filepath = os.path.join(subfolder_path, filename)
                try:
                    # Open the image and ensure it's in RGB format
                    image = Image.open(filepath).convert('RGB').resize(image_size)
                    images.append(np.array(image) / 255.0)  # Normalize pixel values
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {filepath}: {e}")
    images = np.array(images, dtype=np.float32)  # Ensure consistent shape and type
    labels = np.eye(2)[np.array(labels)]  # One-hot encoding for labels
    print(f"Loaded {len(images)} images.")
    return images, labels


class TrainingGUI:
    def __init__(self, cnn_model):
        self.root = tk.Tk()
        self.root.title("CNN Training Visualization")
        self.root.geometry("1920x1080")  # Larger window

        self.cnn_model = cnn_model

        # Pass the GUI's update callback to the CNN model
        self.cnn_model.gui_update_callback = self.gui_update_callback

        # Left frame for loss plot
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right frame for info and architecture
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Training plot
        self.figure, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        self.line, = self.ax.plot([], [], label="Loss", color="blue")
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Training information
        self.info_label = tk.Label(self.info_frame, text="Training Information", font=("Arial", 16, "bold"))
        self.info_label.pack(pady=5)

        self.epoch_label = ttk.Label(self.info_frame, text="Epoch: 0", font=("Arial", 14))
        self.epoch_label.pack(pady=5)

        self.loss_label = ttk.Label(self.info_frame, text="Loss: 0.00", font=("Arial", 14))
        self.loss_label.pack(pady=5)

        self.accuracy_label = ttk.Label(self.info_frame, text="Accuracy: 0.00%", font=("Arial", 14))
        self.accuracy_label.pack(pady=5)

        # Start training button
        self.start_button = ttk.Button(self.info_frame, text="Start Training", command=self.start_training)
        self.start_button.pack(pady=20)

        # Canvas for CNN architecture visualization
        self.arch_label = tk.Label(self.info_frame, text="CNN Architecture", font=("Arial", 16, "bold"))
        self.arch_label.pack(pady=10)

        self.arch_canvas = tk.Canvas(self.info_frame, width=400, height=300, bg="white")
        self.arch_canvas.pack(pady=10)

        # Draw the CNN architecture initially
        self.draw_cnn_architecture()

    def draw_cnn_architecture(self):
        """Visualize the CNN architecture dynamically."""
        self.arch_canvas.delete("all")  # Clear existing drawings
        architecture = [
            {"layer": "Input", "shape": "(64, 64, 3)"},
            {"layer": "Conv2D", "filters": 8, "kernel_size": "(3, 3)", "output_shape": "(62, 62, 8)"},
            {"layer": "Flatten", "output_shape": "(30752)"},
            {"layer": "Dense", "units": 2, "output_shape": "(2)"}
        ]

        x_start = 10
        y_start = 10
        layer_width = 300
        layer_height = 50
        spacing = 20

        for idx, layer in enumerate(architecture):
            layer_name = layer["layer"]
            shape = layer.get("shape", layer.get("output_shape", ""))
            details = f"{layer_name}\n{shape}"

            # Draw rectangle
            self.arch_canvas.create_rectangle(
                x_start, y_start, x_start + layer_width, y_start + layer_height, fill="lightblue", outline="black"
            )
            # Add text inside rectangle
            self.arch_canvas.create_text(
                x_start + layer_width / 2, y_start + layer_height / 2, text=details, font=("Arial", 12)
            )
            # Update y_start for the next layer
            y_start += layer_height + spacing

    def update_architecture(self, current_epoch):
        """Dynamically update the architecture during training."""
        # Example: Highlight a specific layer during the current epoch
        if current_epoch % 2 == 0:
            self.arch_canvas.itemconfig(2, fill="lightgreen")  # Example: Update layer

    def update_plot(self, epoch, loss_history):
        """Update the loss plot."""
        self.line.set_data(range(1, epoch + 1), loss_history)
        self.ax.set_xlim(1, epoch)
        self.ax.set_ylim(0, max(loss_history) * 1.1)
        self.canvas.draw()

    def update_info(self, epoch, loss, accuracy):
        """Update epoch, loss, and accuracy labels."""
        self.epoch_label.config(text=f"Epoch: {epoch}")
        self.loss_label.config(text=f"Loss: {loss:.4f}")
        self.accuracy_label.config(text=f"Accuracy: {accuracy * 100:.2f}%")
        self.update_architecture(epoch)  # Update architecture

    def gui_update_callback(self, epoch, loss, accuracy, losses):
        """Callback for updating the GUI during training."""
        self.update_plot(epoch, losses)
        self.update_info(epoch, loss, accuracy)

    def start_training(self):
        """Start training the CNN."""
        self.start_button.config(state=tk.DISABLED)

        def train_model():
            train_images, train_labels = load_images('data/train')
            self.cnn_model.train_with_visualization(train_images, train_labels, epochs=10, learning_rate=0.01)
            print("Training complete!")

        threading.Thread(target=train_model).start()

    def run(self):
        self.root.mainloop()


# Instantiate and run
cnn_model = CNNWithVisualization(gui_update_callback=None)  # Initialize with placeholder
gui = TrainingGUI(cnn_model)  # Link the actual callback
gui.run()
