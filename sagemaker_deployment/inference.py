import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import base64
from io import BytesIO
import os

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

def model_fn(model_dir):
    device = torch.device('cpu')
    model = DigitClassifier().to(device)
    model_path = os.path.join(model_dir, 'mnist_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def input_fn(request_body, content_type):
    """
    FIX: Do ALL preprocessing here and return tensor ready for model
    """
    if content_type == 'application/json':
        # Parse JSON
        if isinstance(request_body, bytes):
            request_body = request_body.decode('utf-8')
        data = json.loads(request_body)
        image_data = base64.b64decode(data['image'])

        # Open image from bytes
        img = Image.open(BytesIO(image_data)).convert('L')

        # Preprocess (same as before)
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

        # Convert to tensor - RETURN THIS
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        img_tensor = transform(canvas).unsqueeze(0)
        return img_tensor
    else:
        raise ValueError(f'Unsupported content type: {content_type}')

def predict_fn(data, model):
    """
    FIX: data is already a tensor, just run model
    """
    with torch.no_grad():
        output = model(data)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()

    return prediction, confidence

def output_fn(prediction, accept):
    pred, conf = prediction
    return json.dumps({
        'prediction': int(pred),
        'confidence': round(float(conf) * 100, 2)
    }), 'application/json'

