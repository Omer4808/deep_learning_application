import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
import io
import os

# Define class labels for Fashion MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
MODEL_PATH = "./model/fashion_mnist_best_model_weights.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model (must match the saved model architecture)
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)

        self._to_linear = None
        self._compute_linear_input()

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 10)

    def _compute_linear_input(self):
        """Compute the number of flattened features dynamically."""
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, 28, 28)
            output = self.pool(torch.relu(self.conv1(sample_input)))
            output = self.dropout(output)
            output = self.pool(torch.relu(self.conv2(output)))
            output = self.dropout(output)
            # Apply pooling after conv3
            output = self.pool(torch.relu(self.conv3(output)))  
            output = self.dropout(output)
            # Get the total number of features
            self._to_linear = output.numel()  

    def forward(self, x):
        """
        Forward pass of the CNN model.

        The model applies a series of convolutional and fully connected layers.
        The convolutional layers are composed of a convolutional layer,
        a ReLU activation function, a max pooling layer with a kernel size of 2,
        and a dropout layer with a dropout rate of 0.5.

        The fully connected layers are composed of a linear layer with a ReLU
        activation function and a dropout layer with a dropout rate of 0.5.

        :param x: The input tensor.
        :return: The output tensor.
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

app = Flask(__name__)


model = FashionCNN().to(device)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print(f"❌ Model file not found at {MODEL_PATH}")
    model = None

# Define image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Please check the server logs."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Empty file uploaded"}), 400

        image = Image.open(io.BytesIO(image_bytes))

        # Ensure the image is in RGB or grayscale mode
        if image.mode not in ["RGB", "L"]:
            # Convert to grayscale if needed
            image = image.convert("L")  

        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            label = class_labels[predicted.item()]

        print(f"🟢 Prediction: {label}")
        return jsonify({"prediction": label})

    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image format"}), 400

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
