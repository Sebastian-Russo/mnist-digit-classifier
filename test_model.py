import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Same model architecture as training
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_model(model_path='mnist_model.pth'):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def preprocess_image(image_path):
    """Preprocess an image for prediction"""
    # Open image and convert to grayscale
    img = Image.open(image_path).convert('L')

    # Resize to 28x28
    img = img.resize((28, 28))

    # Convert to tensor and normalize (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img_tensor = transform(img)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def predict(model, device, image_path):
    """Make a prediction on a single image"""
    # Preprocess image
    img_tensor = preprocess_image(image_path).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_digit = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_digit].item()

    # Get all probabilities
    all_probs = probabilities[0].cpu().numpy()

    return predicted_digit, confidence, all_probs

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_model.py <image_path>")
        print("Example: python test_model.py digit.png")
        return

    image_path = sys.argv[1]

    # Load model
    print("Loading model...")
    model, device = load_model()

    # Make prediction
    print(f"Predicting digit in {image_path}...")
    predicted_digit, confidence, all_probs = predict(model, device, image_path)

    print(f"\nPredicted digit: {predicted_digit}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll probabilities:")
    for digit, prob in enumerate(all_probs):
        print(f"  {digit}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()
