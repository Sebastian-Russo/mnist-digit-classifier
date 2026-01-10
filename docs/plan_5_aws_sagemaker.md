cd ~/ai-projects/mnist-digit-classifier

cat > AWS_SAGEMAKER_COMPLETE_GUIDE.md << 'EOF'
# AWS SageMaker Deployment - Complete Guide

## Why SageMaker?

Lambda couldn't handle PyTorch (250MB unzipped limit). SageMaker is AWS's purpose-built ML service with no size limits.

---

## What You're Replacing & How It All Connects

### The Journey So Far

| Stage | What You Built | Purpose | Files Created |
|-------|----------------|---------|---------------|
| **Phase 1: Training** | CNN model | Train the "brain" | `train_mnist.py`, `mnist_model.pth` |
| **Phase 2: Local Prediction** | Standalone script | Test locally | `predict.py`, `visualize.py` |
| **Phase 3: Django API** | Local web API | Make it accessible via HTTP | `views.py`, `model_loader.py`, `urls.py` |
| **Phase 4: React Frontend** | Web interface | User can draw digits | `App.js` |
| **Phase 5: Lambda** | ❌ Failed | PyTorch too large (250MB limit) | `lambda_function.py` |
| **Phase 6: SageMaker** | ✅ Production ML | Cloud deployment that works | `inference.py` |

---

## Architecture Evolution: How Components Map

### Phase 1: MNIST Training (Local)
```
train_mnist.py
    ↓ Trains model
mnist_model.pth (1MB saved weights)
```
**Purpose:** Create the trained "brain"
**Files:** `train/train_mnist.py`, `mnist_model.pth`

---

### Phase 2: Local Prediction Script
```
predict.py
    ↓ Loads
mnist_model.pth
    ↓ Preprocesses & predicts
Result
```
**Purpose:** Test model works
**Files:** `predict/predict.py`
**Key functions:**
- `preprocess_image()` - Clean up image
- `predict_digit()` - Make prediction

---

### Phase 3: Django API (Local Web Server)
```
React Frontend (http://localhost:3000)
    ↓ POST /predict with image
Django Server (http://localhost:8000)
    ↓ urls.py routes to views.py
views.predict_digit()
    ↓ Calls
model_loader.predict()
    ↓ Uses
mnist_model.pth
    ↓ Returns
{"prediction": 5, "confidence": 94.67}
```
**Purpose:** Make model accessible via HTTP locally
**Files:**
- `api/classifier/views.py` - HTTP request handler (like a waiter)
- `api/classifier/model_loader.py` - Model loading + prediction (like a chef)
- `api/classifier/urls.py` - Route mapping (like a menu)
- `api/api/settings.py` - Django config

**Key Django components:**
- **views.py** = Request handler ("waiter takes order")
- **model_loader.py** = Business logic ("chef cooks food")
- **urls.py** = Routing ("menu shows what's available")

---

### Phase 4: React Frontend
```
User draws digit in browser
    ↓
React App.js
    ↓ Converts canvas to base64
    ↓ POST request
Django API or Lambda or SageMaker
```
**Purpose:** User-friendly interface
**Files:** `frontend/src/App.js`
**Key code:**
```javascript
fetch('http://localhost:8000/predict/', {
  method: 'POST',
  body: formData  // or JSON with base64
})
```

---

### Phase 5: Lambda Attempt (FAILED)
```
React → API Gateway → Lambda → ❌ Can't fit PyTorch
```
**Why it failed:**
- Lambda limit: 250MB unzipped
- PyTorch alone: 250MB+ unzipped
- Math doesn't work

**Files created (not used):**
- `lambda_function/lambda_function.py` - Handler (like views.py)
- `lambda_function/model_loader.py` - Modified for S3

---

### Phase 6: SageMaker (PRODUCTION)
```
React Frontend
    ↓ POST with base64 image
API Gateway (optional)
    ↓
SageMaker Endpoint
    ├── inference.py (replaces views.py + model_loader.py)
    ├── mnist_model.pth
    └── PyTorch container (managed by AWS)
    ↓
{"prediction": 5, "confidence": 94.67}
```
**Purpose:** Production-ready ML deployment
**Files:** `sagemaker_deployment/inference.py`, `model.tar.gz`

---

## How SageMaker Maps to Django

| Django Component | SageMaker Component | What It Does |
|------------------|---------------------|--------------|
| **manage.py runserver** | SageMaker Endpoint | Starts the service |
| **urls.py** | Built-in (handled by container) | Routes requests |
| **views.predict_digit()** | input_fn() + output_fn() | Parse request, format response |
| **model_loader.py** | model_fn() + predict_fn() | Load model, make predictions |
| **settings.py** | Endpoint configuration | Memory, instance type, etc. |
| **Django app running** | Endpoint "InService" | Ready to receive requests |

---

## Code Comparison: Django vs SageMaker

### Django Structure
```python
# model_loader.py
model = DigitClassifier()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

def predict(image_file):
    # Preprocess
    processed = preprocess_image(image_file)
    # Predict
    output = model(processed)
    return prediction, confidence

# views.py
def predict_digit(request):
    image = request.FILES['image']
    prediction, confidence = predict(image)
    return JsonResponse({'prediction': prediction, 'confidence': confidence})
```

### SageMaker Structure
```python
# inference.py (combines both files)
def model_fn(model_dir):
    # Load model (runs once at startup)
    model = DigitClassifier()
    model.load_state_dict(torch.load(os.path.join(model_dir, 'mnist_model.pth')))
    return model

def input_fn(request_body, content_type):
    # Parse request (like views.py parsing request.FILES)
    data = json.loads(request_body)
    return base64.b64decode(data['image'])

def predict_fn(data, model):
    # Preprocess and predict (like model_loader.predict)
    processed = preprocess_image(data)
    output = model(processed)
    return prediction, confidence

def output_fn(prediction, accept):
    # Format response (like views.py JsonResponse)
    return json.dumps({'prediction': pred, 'confidence': conf})
```

**Key insight:** SageMaker combines views.py + model_loader.py into one file with specific function names.

---

## Step 1: Prepare Model for SageMaker

### 1.1: Create inference script

**Directory:** Project root
```bash
cd ~/ai-projects/mnist-digit-classifier
mkdir sagemaker_deployment
cd sagemaker_deployment
```

**Create `inference.py`:**

This file replaces BOTH `model_loader.py` AND `views.py`:
```python
"""
SageMaker inference script
Combines Django's views.py + model_loader.py into one file

Function mapping:
- model_fn() → model_loader.py initialization
- input_fn() → views.py request parsing
- predict_fn() → model_loader.predict()
- output_fn() → views.py JsonResponse
"""
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import base64
from io import BytesIO
import os

# CNN Architecture (SAME as train_mnist.py)
class DigitClassifier(nn.Module):
    """
    Same architecture you trained in Phase 1
    Must match exactly or model weights won't load
    """
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

# SageMaker required functions (replaces Django components)

def model_fn(model_dir):
    """
    REPLACES: model_loader.py initialization
    DJANGO EQUIVALENT: The code that runs when Django starts

    Called ONCE when endpoint starts (like Django loading model at startup)
    """
    device = torch.device('cpu')
    model = DigitClassifier().to(device)
    model_path = os.path.join(model_dir, 'mnist_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def input_fn(request_body, content_type):
    """
    REPLACES: views.py request parsing
    DJANGO EQUIVALENT: request.FILES['image'] or request.body

    Parses incoming HTTP request
    """
    if content_type == 'application/json':
        # Handle both string and bytes
        if isinstance(request_body, bytes):
            request_body = request_body.decode('utf-8')

        data = json.loads(request_body)
        image_data = base64.b64decode(data['image'])
        return BytesIO(image_data)
    else:
        raise ValueError(f'Unsupported content type: {content_type}')

def predict_fn(data, model):
    """
    REPLACES: model_loader.predict() and preprocess_image()
    DJANGO EQUIVALENT: The actual prediction logic

    Same preprocessing as predict.py and model_loader.py
    """
    # Preprocess image (SAME as predict.py)
    img = Image.open(data).convert('L')
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

    # Convert to tensor (SAME as train_mnist.py and predict.py)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img_tensor = transform(canvas).unsqueeze(0)

    # Predict (SAME logic as everywhere else)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()

    return prediction, confidence

def output_fn(prediction, accept):
    """
    REPLACES: views.py JsonResponse
    DJANGO EQUIVALENT: return JsonResponse({...})

    Formats response as JSON
    """
    pred, conf = prediction
    return json.dumps({
        'prediction': int(pred),
        'confidence': round(float(conf) * 100, 2)
    }), 'application/json'
```

### 1.2: Package model for SageMaker

**Directory:** `sagemaker_deployment/`
```bash
cd ~/ai-projects/mnist-digit-classifier/sagemaker_deployment

# Copy the model you trained in Phase 1
cp ../mnist_model.pth .

# Create tar.gz (SageMaker requirement - like zipping for deployment)
tar -czf model.tar.gz mnist_model.pth inference.py

# Upload to S3 (like deploying to a server)
aws s3 cp model.tar.gz s3://mnist-model-ai-training-sebastian/sagemaker/ --profile firstfire
```

**What just happened:**
- `model.tar.gz` contains your trained model + inference code
- Uploaded to S3 so SageMaker can access it
- Like pushing code to a server, but for ML models

---

## Step 2: Create SageMaker Endpoint

### Critical Issue: Console BYOC vs Script Mode

**Problem:**
The AWS SageMaker Console's "Provide model artifacts and inference image location" creates the model in **BYOC (Bring Your Own Container)** mode, which **ignores your `inference.py`**. The container falls back to its default handler, causing the UnicodeDecodeError you saw.

**Why This Happens:**
- BYOC mode expects a custom Docker container, not a script.
- Your `inference.py` is never called; the container tries to decode raw bytes as UTF-8.
- **Script Mode** (which uses your `inference.py`) is only available via:
  - SageMaker Python SDK
  - SageMaker Notebook

**Solution:**
Use a SageMaker Notebook to deploy your model in Script Mode. This is the simplest path without writing SDK code.

---

## Step 2 (Revised): Deploy Using SageMaker Notebook

### 2.1: Create SageMaker Notebook Instance

1. Go to **SageMaker** console
2. Left sidebar: **Notebook** → **Notebook instances**
3. Click **Create notebook instance**
   - **Notebook instance name:** `mnist-deployment`
   - **Instance type:** `ml.t2.medium` (or your choice)
   - **IAM role:** Select or create a role with SageMaker and S3 access
4. Click **Create notebook instance**
5. Wait for status: **InService**
6. Click **Open Jupyter** next to your notebook instance

### 2.1.1: Best Practice IAM Setup

**For development/experimentation in notebook instances, use AWS managed policy:**

1. Go to **IAM** → **Roles** → `SageMaker-mnist-model-ai-training-sebastian`
2. **Add permissions** → **Attach policies**
3. Search for and attach **AmazonSageMakerFullAccess**
4. **Stop and restart** the notebook instance (required for new permissions to take effect)

**Why this is better:**
- Includes all SageMaker actions (CreateModel, CreateEndpoint, etc.)
- Includes necessary S3/ECR/CloudWatch/Logs permissions
- Includes `iam:GetRole` permission (fixes the get_role error)
- No custom policy maintenance needed

### 2.2: Upload and Deploy Model in Notebook

1. In JupyterLab, create a new notebook:
   - In the top menu, click **File** → **New** → **Notebook**.
   - In the "Select Kernel" dialog, choose **conda_python3** (or any Python 3 kernel).
2. In the first code cell, download your `model.tar.gz`:
   ```python
   !aws s3 cp s3://mnist-model-ai-training-sebastian/sagemaker/model.tar.gz .
   ```
3. In the next cell, deploy using SageMaker SDK (built-in to notebooks):
   ```python
   import sagemaker
   from sagemaker.pytorch.model import PyTorchModel

   role = sagemaker.get_execution_role()
   model = PyTorchModel(
       model_data='s3://mnist-model-ai-training-sebastian/sagemaker/model.tar.gz',
       role=role,
       entry_point='inference.py',
       framework_version='2.1.0',
       py_version='py310',
       source_dir=None  # inference.py is at root of tar.gz
   )

   predictor = model.deploy(
       initial_instance_count=1,
       instance_type='ml.t2.medium',
       endpoint_name='mnist-classifier-endpoint'
   )
   ```
4. Run the cell. This will:
   - Create the model in **Script Mode** (your `inference.py` will be used)
   - Create endpoint configuration
   - Deploy the endpoint
   - Wait until **InService**

### 2.3: Verify Deployment

After deployment succeeds, you can delete the notebook instance if you only need the endpoint.

---

## Step 3: Test SageMaker Endpoint

### 3.1: Test via CLI

**Directory:** Project root
```bash
cd ~/ai-projects/mnist-digit-classifier

# Use the same test file from Django testing
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name mnist-classifier-endpoint \
  --content-type application/json \
  --body file://test_payload.json \
  --region us-east-1 \
  --profile firstfire \
  output.json

cat output.json
```

**Expected response (SAME as Django):**
```json
{"prediction": 5, "confidence": 94.67}
```

---

## Common Issue: Base64 Padding Error

### Problem
When testing the endpoint, you may get:
```
ModelError: Received server error (500) from primary with message "Incorrect padding"
```

### Why This Happens
- Base64 strings must have a length that's a multiple of 4
- When copied/pasted, the padding characters (`=`) at the end may be lost
- SageMaker's `base64.b64decode()` is strict about padding

### Fix in Notebook
```python
import json
import boto3
import base64

# Read your test file
with open('test_payload.json', 'r') as f:
    test_payload = json.load(f)

# Fix padding
base64_str = test_payload['image']
padding = len(base64_str) % 4
if padding:
    base64_str += '=' * (4 - padding)
    test_payload['image'] = base64_str

# Test
runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='mnist-classifier-endpoint',
    ContentType='application/json',
    Body=json.dumps(test_payload)
)

result = json.loads(response['Body'].read())
print(f"Prediction: {result}")
```

This ensures the base64 string is always valid before sending to the endpoint.

---

## Common Issue: PIL Truncated Image Error

### Problem
After fixing base64 padding, you may get:
```
OSError: Truncated File Read
```
in `predict_fn` at line: `img = Image.open(data).convert('L')`

### Why This Happens
- The base64 string is decoded successfully but the resulting bytes form an incomplete/truncated PNG
- PIL/Pillow detects the PNG file is incomplete during parsing
- Common causes:
  - Base64 string was truncated before encoding
  - Image data was cut off during copy-paste
  - Network/JSON encoding dropped bytes

### Fix 1: Allow Truncated Images (Quick Fix)
Add to `inference.py` after imports:
```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

### Fix 2: Add Debug Logging
Add to `inference.py` after imports:
```python
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

Update `input_fn` after decode:
```python
image_data = base64.b64decode(data['image'])
logger.debug(f"Decoded bytes length: {len(image_data)}")
logger.debug(f"First 8 bytes: {image_data[:8].hex()}")  # expect 89504e470d0a1a0a
```

Update `predict_fn` around Image.open:
```python
try:
    img = Image.open(data).convert('L')
    logger.info(f"Image opened: mode={img.mode}, size={img.size}")
except Exception as e:
    logger.exception("Image open failed")
    raise
```

### Re-deploy Steps
```python
# Re-package with updated inference.py
!tar -czf model.tar.gz mnist_model.pth inference.py

# Upload and update endpoint
!aws s3 cp model.tar.gz s3://sagemaker-us-east-1-308665918648/model.tar.gz

# Update endpoint with update_endpoint=True
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    endpoint_name='mnist-classifier-endpoint',
    update_endpoint=True
)
```

### Check CloudWatch Logs
After testing, check CloudWatch logs for debug output:
- Valid PNG: first 8 bytes = `89504e470d0a1a0a`
- MNIST PNG length: ~1-5 KB
- If length < 100 bytes or signature wrong → base64 is truncated

---
## Checkpoint
## Current Status: Testing Model Endpoint

**Last Updated:** 2026-01-10 2:03 PM

**Issue:** Attempting to test the deployed SageMaker endpoint with a base64-encoded image but encountering string formatting issues in the notebook.

**Where We Left Off:**
1. Successfully deployed the PyTorch model to SageMaker endpoint `mnist-classifier-endpoint`
2. Fixed base64 padding issue in previous test
3. Now encountering `SyntaxError: unterminated string literal` when trying to use a verified MNIST test image
4. The base64 string is too long and getting split across lines in the notebook

**Next Steps:**
- [ ] Use a shorter test approach or load from file
- [ ] Successfully invoke the endpoint and get a prediction
- [ ] Verify the model is working correctly
- [ ] Document the final working test code

**Current Error:**
```python
SyntaxError: unterminated string literal (detected at line 6)
```
When trying to use a long base64 string in the notebook test cell.

---

## Step 4: Update React Frontend

### 4.1: Direct SageMaker endpoint (for testing)

**Directory:** `frontend/src/`
```bash
cd ~/ai-projects/mnist-digit-classifier/frontend/src
```

Open `App.js` and update the fetch URL.

**Find:**
```javascript
const response = await fetch('http://localhost:8000/predict/', {
```

**Replace with:**
```javascript
// Direct SageMaker endpoint (requires AWS credentials - for testing only)
const response = await fetch('https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/mnist-classifier-endpoint/invocations', {
```

**Note:** This won't work from browser due to CORS and authentication. See Step 5 for production setup.

---

## Step 5: (Recommended) Add API Gateway

For production, add API Gateway + Lambda (like Django routing):
```
React → API Gateway → Lambda → SageMaker Endpoint
```

**Lambda code (like a thin views.py):**
```python
import boto3
import json

sagemaker = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    """
    Like Django views.py but just forwards to SageMaker
    Lambda CAN handle this because it's just a proxy, not running PyTorch
    """
    body = json.loads(event['body'])

    # Call SageMaker endpoint
    response = sagemaker.invoke_endpoint(
        EndpointName='mnist-classifier-endpoint',
        ContentType='application/json',
        Body=json.dumps(body)
    )

    result = json.loads(response['Body'].read())

    return {
        'statusCode': 200,
        'headers': {'Access-Control-Allow-Origin': '*'},
        'body': json.dumps(result)
    }
```

Then React calls API Gateway URL (same as Django setup).

---

## Cost Management

### Delete Resources When Not Using

**Stop endpoint (like stopping Django server):**
```bash
aws sagemaker delete-endpoint --endpoint-name mnist-classifier-endpoint --profile firstfire
```

**Cost while running:**
- ml.t2.medium: ~$0.065/hour (~$47/month if always on)

**For learning:**
- Create endpoint when testing ($0.50 for few hours)
- Delete when done
- Total cost: $1-5 for testing

---

## Summary: What You've Built

| Phase | Technology | Purpose | Status | Files |
|-------|------------|---------|--------|-------|
| **1** | PyTorch | Train model | ✅ | `train_mnist.py`, `mnist_model.pth` |
| **2** | Python script | Test locally | ✅ | `predict.py` |
| **3** | Django | Local web API | ✅ | `views.py`, `model_loader.py` |
| **4** | React | User interface | ✅ | `App.js` |
| **5** | Lambda | Cloud deployment | ❌ Too large | `lambda_function.py` |
| **6** | SageMaker | Production ML | ✅ | `inference.py` |

---

## Key Learnings

1. **Same model throughout** - `mnist_model.pth` created in Phase 1 is used everywhere
2. **Same preprocessing** - Code from `predict.py` → `model_loader.py` → `inference.py`
3. **Architecture patterns repeat** - Request → Handler → Business Logic → Response
4. **Django taught you to pattern** - SageMaker is just a different implementation
5. **Lambda failed but taught you limits** - Not every tool fits every job
6. **Console limitation** - BYOC mode ignores your script; use Notebook or SDK for Script Mode

**You now understand production ML deployment!** 🎉

EOF

echo "✅ Created AWS_SAGEMAKER_COMPLETE_GUIDE.md"
echo ""
echo "This guide shows:"
echo "- How all phases connect (training → predict → Django → React → SageMaker)"
echo "- Code comparisons (Django vs SageMaker)"
echo "- What each file replaces"
echo "- Complete step-by-step with context"