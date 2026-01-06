"""
DETECTIVE AGENCY ANALOGY:
This file is the agency's PERMANENT OFFICE that stays open 24/7.

WHY THIS FILE EXISTS:
- Your train_mnist.py: Built and trained the detective agency (one-time setup)
- Your predict.py: Stand-alone script that reopens the agency each time you run it
- THIS FILE: Keeps the agency ALWAYS OPEN for the restaurant (Django) to use

When customers order from the restaurant (Django API):
- Restaurant doesn't train detectives (that's already done)
- Restaurant just asks the OPEN agency to solve cases
- This file = The agency's front door that's always unlocked
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os

# ANALOGY: The detective agency blueprint (same as training)
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 32 junior detectives
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 senior detectives
        self.pool = nn.MaxPool2d(2, 2)  # Assistant - Junior detectives write detailed reports → Assistant keeps only the strongest findings from each section
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 128 analysts
        self.fc2 = nn.Linear(128, 10)  # 10 verdict specialists
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ANALOGY: Open the detective agency and keep it running
# (Happens once when Django starts, not every request!)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitClassifier().to(device)
model_path = os.path.join(os.path.dirname(__file__), 'mnist_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))  # Load trained experience
model.eval()  # Agency is open for business!

def preprocess_image(image_file):
    """
    ANALOGY: Case file preparation department
    Cleans up messy evidence photos before detectives see them
    """
    img = Image.open(image_file).convert('L')
    img = ImageOps.invert(img)

    img_array = np.array(img)
    img_array = np.where(img_array > 30, 255, 0).astype(np.uint8)

    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img_array = img_array[rmin:rmax+1, cmin:cmax+1]

    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.MaxFilter(3))

    width, height = img.size
    if width > height:
        new_width = 20
        new_height = max(int(height * 20 / width), 1)
    else:
        new_height = 20
        new_width = max(int(width * 20 / height), 1)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new('L', (28, 28), color=0)
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    canvas.paste(img, (paste_x, paste_y))

    return canvas

def predict(image_file):
    """
    ANALOGY: Process one case through the agency
    (Restaurant calls this function when customer orders)
    """
    # Prep department cleans the evidence
    processed_img = preprocess_image(image_file)

    # Format for agency standards
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img_tensor = transform(processed_img).unsqueeze(0).to(device)

    # Send through the entire detective agency
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()

    # Return the verdict
    return prediction, confidence