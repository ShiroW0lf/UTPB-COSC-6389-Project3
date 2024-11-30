import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


# Simulated CNN Class
class CNNWithVisualization:
    def __init__(self, gui_update_callback):
        self.gui_update_callback = gui_update_callback

    def train_with_visualization(self, train_images, train_labels, epochs=10, learning_rate=0.01):
        losses = []
        for epoch in range(1, epochs + 1):
            # Simulate training process
            loss = np.random.uniform(0.1, 0.5) / epoch
            accuracy = 0.5 + np.random.uniform(0.01, 0.1) * epoch / epochs

            losses.append(loss)
            print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

            # Update GUI
            self.gui_update_callback(epoch, loss, accuracy, losses)

    @staticmethod
    def predict(images):
        return np.random.randint(0, 2, size=len(images))


# Simulated Image Loader
def load_images(directory):
    # Simulating dataset loading
    num_samples = 100
    image_size = (64, 64, 3)
    images = np.random.random((num_samples, *image_size))
    labels = np.random.randint(0, 2, size=num_samples)
    return images, labels


# Training GUI
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

        # Start Training Button
        self.start_button = ttk.Button(self.info_frame, text="Start Training", command=self.start_training)
        self.start_button.pack(pady=20)

    def update_plot(self, epoch, loss_history):
        self.line.set_data(range(1, epoch + 1), loss_history)
        self.ax.set_xlim(1, epoch)
        self.ax.set_ylim(0, max(loss_history) * 1.1)
        self.canvas.draw()

    def update_info(self, epoch, loss, accuracy):
        self.epoch_label.config(text=f"Epoch: {epoch}")
        self.loss_label.config(text=f"Loss: {loss:.4f}")
        self.accuracy_label.config(text=f"Accuracy: {accuracy * 100:.2f}%")

    def gui_update_callback(self, epoch, loss, accuracy, losses):
        self.update_plot(epoch, losses)
        self.update_info(epoch, loss, accuracy)

    def start_training(self):
        # Disable the button to prevent multiple training sessions
        self.start_button.config(state=tk.DISABLED)

        # Start training in a separate thread to keep the UI responsive
        def train_model():
            train_images, train_labels = load_images('data/train')
            self.cnn_model.train_with_visualization(train_images, train_labels, epochs=10, learning_rate=0.01)

        training_thread = threading.Thread(target=train_model)
        training_thread.start()

    def run(self):
        self.root.mainloop()


# Instantiate CNN and GUI
def gui_update_callback(epoch, loss, accuracy, losses):
    gui.update_plot(epoch, losses)
    gui.update_info(epoch, loss, accuracy)


cnn_model = CNNWithVisualization(gui_update_callback)
gui = TrainingGUI(cnn_model)

# Start the GUI
gui.run()
