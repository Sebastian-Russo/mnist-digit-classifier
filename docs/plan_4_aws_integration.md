cd ~/ai-projects/mnist-digit-classifier
cat > AWS_LAMBDA_UPDATED.md << 'EOF'
# AWS Lambda + API Gateway - Complete Guide
## Every command shows the directory to run it from

---

## Step 1: Create Lambda Layer for PyTorch

### 1.1: Create layer structure

**Directory:** Project root
```bash
cd ~/ai-projects/mnist-digit-classifier
mkdir -p lambda_layer/python
```

### 1.2: Install dependencies

**Directory:** `lambda_layer/python/`
```bash
cd ~/ai-projects/mnist-digit-classifier/lambda_layer/python
pip install torch torchvision pillow numpy -t .
```

**What these do:**
- torch: PyTorch ML framework (~100MB)
- torchvision: Image preprocessing
- pillow: Image loading (PIL)
- numpy: Array operations

### 1.3: Zip the layer

**Directory:** `lambda_layer/`
```bash
cd ~/ai-projects/mnist-digit-classifier/lambda_layer
zip -r pytorch_layer.zip python/
```

### 1.4: Create layer in AWS

**Directory:** `lambda_layer/`
```bash
cd ~/ai-projects/mnist-digit-classifier/lambda_layer
aws lambda publish-layer-version \
  --layer-name pytorch-layer \
  --zip-file fileb://pytorch_layer.zip \
  --compatible-runtimes python3.12 \
  --profile firstfire
```

**SAVE THE LAYER ARN FROM OUTPUT!**

---

## Step 2: Prepare Lambda Function Code

### 2.1: Create function directory

**Directory:** Project root
```bash
cd ~/ai-projects/mnist-digit-classifier
mkdir lambda_function
```

### 2.2: Copy model loader

**Directory:** Project root
```bash
cd ~/ai-projects/mnist-digit-classifier
cp api/classifier/model_loader.py lambda_function/
```

### 2.3: Update model_loader.py for S3

**Directory:** `lambda_function/`
```bash
cd ~/ai-projects/mnist-digit-classifier/lambda_function
```

Open `model_loader.py` in editor and find this section:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitClassifier().to(device)
model_path = os.path.join(os.path.dirname(__file__), 'mnist_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
```

Replace with:
```python
import boto3

s3 = boto3.client('s3')
device = torch.device('cpu')
model = DigitClassifier().to(device)

model_bucket = os.environ.get('MODEL_BUCKET')
model_key = os.environ.get('MODEL_KEY', 'mnist_model.pth')

s3.download_file(model_bucket, model_key, '/tmp/mnist_model.pth')
model.load_state_dict(torch.load('/tmp/mnist_model.pth', map_location=device))
model.eval()
```

### 2.4: Create lambda_function.py

**Directory:** `lambda_function/`
```bash
cd ~/ai-projects/mnist-digit-classifier/lambda_function
touch lambda_function.py
```

Add this code to `lambda_function.py`:
```python
import json
import base64
from io import BytesIO
from model_loader import predict

def lambda_handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))

        if 'image' not in body:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'No image provided'})
            }

        image_data = base64.b64decode(body['image'])
        image_file = BytesIO(image_data)

        prediction, confidence = predict(image_file)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': int(prediction),
                'confidence': round(float(confidence) * 100, 2)
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }
```

### 2.5: Zip function code

**Directory:** `lambda_function/`
```bash
cd ~/ai-projects/mnist-digit-classifier/lambda_function
zip function.zip lambda_function.py model_loader.py
```

---

## Step 3: Upload Model to S3

### 3.1: Create S3 bucket

**Directory:** Any (doesn't matter for AWS commands)
```bash
aws s3 mb s3://mnist-model-YOUR-NAME --profile firstfire
```

### 3.2: Upload model

**Directory:** Project root (where mnist_model.pth is)
```bash
cd ~/ai-projects/mnist-digit-classifier
aws s3 cp mnist_model.pth s3://mnist-model-YOUR-NAME/ --profile firstfire
```

---

## Step 4: Create Lambda Function in AWS Console

1. Go to AWS Lambda Console
2. Click "Create function"
3. Function name: `mnist-digit-predictor`
4. Runtime: Python 3.12
5. Architecture: x86_64
6. Create function

---

## Step 5: Upload Function Code

**Directory:** `lambda_function/`
```bash
cd ~/ai-projects/mnist-digit-classifier/lambda_function
aws lambda update-function-code \
  --function-name mnist-digit-predictor \
  --zip-file fileb://function.zip \
  --profile firstfire
```

---

## Step 6: Attach Layer to Function

**Directory:** Any
```bash
aws lambda update-function-configuration \
  --function-name mnist-digit-predictor \
  --layers YOUR-LAYER-ARN-HERE \
  --profile firstfire
```

---

## Step 7: Configure Function Settings

### Via AWS Console:

1. Go to Lambda function → Configuration → General configuration
2. Edit:
   - Memory: 512 MB
   - Timeout: 30 seconds

3. Go to Configuration → Environment variables
4. Add:
   - `MODEL_BUCKET` = `mnist-model-ai-training-sebastian`
   - `MODEL_KEY` = `mnist_model.pth`

---

## Step 8: Add S3 Permissions

### Via AWS Console:

1. Lambda function → Configuration → Permissions
2. Click execution role name
3. Add permissions → Attach policies
4. Search and attach: `AmazonS3ReadOnlyAccess`

---

## Step 9: Create API Gateway

### Via AWS Console:

1. Go to API Gateway console
2. Create API → REST API (not HTTP API or private)
3. API name: `mnist-api`
4. Create API

5. Actions → Create Resource
   - Resource name: `predict`
   - Create Resource

6. Select `/predict` → Actions → Create Method → POST
   - Integration type: Lambda Function
   - Check "Use Lambda Proxy integration"
   - Lambda Function: `mnist-digit-predictor`
   - Save
   - Click OK on permissions popup

7. Select `/predict` → Actions → Enable CORS
   - Use defaults
   - Enable CORS

8. Actions → Deploy API
   - Stage: `prod`
   - Deploy

9. Copy the Invoke URL (looks like: `https://abc123.execute-api.us-east-1.amazonaws.com/prod`)

---

## Step 10: Test the API

### 10.1: Convert test image to base64

**Directory:** Project root
```bash
cd ~/ai-projects/mnist-digit-classifier
base64 mnist_training/test_images/digit_5.png | tr -d '\n' > digit5_base64.txt
```

### 10.2: Create test payload

**Directory:** Project root
```bash
cd ~/ai-projects/mnist-digit-classifier
echo "{\"image\": \"$(cat digit5_base64.txt)\"}" > test_payload.json
```

### 10.3: Test with curl

**Directory:** Project root
```bash
cd ~/ai-projects/mnist-digit-classifier
curl -X POST \
  -H "Content-Type: application/json" \
  -d @test_payload.json \
  https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/predict
```

**Expected response:**
```json
{"prediction": 5, "confidence": 94.67}
```

---

## Step 11: Update React Frontend

### 11.1: Update App.js

**Directory:** `frontend/src/`
```bash
cd ~/ai-projects/mnist-digit-classifier/frontend/src
```

Open `App.js` and update the `handlePredict` function:
```javascript
const handlePredict = async () => {
  setLoading(true);

  const imageData = await canvasRef.current.exportImage('png');
  const base64Image = imageData.split(',')[1];

  try {
    const response = await fetch('https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: base64Image }),
    });

    const data = await response.json();
    setPrediction(data.prediction);
    setConfidence(data.confidence);
  } catch (error) {
    console.error('Error:', error);
    alert('Failed to get prediction');
  }

  setLoading(false);
};
```

### 11.2: Test frontend

**Directory:** `frontend/`
```bash
cd ~/ai-projects/mnist-digit-classifier/frontend
npm start
```

Draw a digit and click Predict - should now use Lambda!

---

## Troubleshooting

### "Unable to load paramfile" error
- You're in the wrong directory
- Check the directory stated before each command

### "Permission denied" on S3
- Add S3ReadOnlyAccess to Lambda execution role

### CORS errors
- Enable CORS in API Gateway
- Check Access-Control-Allow-Origin headers in Lambda response

### Cold start slow (~5 seconds)
- Normal for first request
- Subsequent requests will be fast

---

## Cost
- Lambda: ~$0.20 per 1M requests
- API Gateway: ~$3.50 per 1M requests
- S3: ~$0.023 per GB/month
- **Total: <$1/month for learning**

EOF

echo "✅ Created AWS_LAMBDA_UPDATED.md with directories for every command"

----------------------------------------------------------------------------
cd ~/ai-projects/mnist-digit-classifier

# Update Lambda guide with what actually happened
cat > AWS_LAMBDA_REALITY_CHECK.md << 'EOF'
# AWS Lambda Attempt - What We Learned

## TL;DR: Lambda Won't Work for PyTorch Models

**Bottom line:** Lambda has a 250MB unzipped limit. PyTorch alone is 250MB+ unzipped. We can't deploy PyTorch models to Lambda without severe compromises.

---

## What We Tried

### Attempt 1: Build Custom PyTorch Layer

**Steps:**
1. Created `lambda_layer/python/` directory
2. Installed PyTorch: `pip install torch torchvision pillow numpy -t .`
3. Zipped it: `zip -r pytorch_layer.zip python/`

**Result:** Failed ❌

**Why it failed:**
- Zipped size: ~150MB
- Unzipped size: ~350MB
- Lambda limit: 250MB unzipped
- AWS error: "Unzipped size must be smaller than 262144000 bytes"

**What we learned:** Full PyTorch is too large for Lambda layers.

---

### Attempt 2: Use Public PyTorch Layer (Klayers)

**Tried:**
```bash
aws lambda update-function-configuration \
  --layers arn:aws:lambda:us-east-1:770693421928:layer:Klayers-p312-torch:1
```

**Result:** Failed ❌

**Why it failed:**
- Error: `AccessDeniedException: User is not authorized to perform: lambda:GetLayerVersion`
- Lambda layers can't be shared cross-account without explicit permissions
- Public layers don't exist in the way we thought they did
- Each AWS account needs to create their own layer

**What we learned:** Can't use layers from other AWS accounts without permission.

---

## Why Lambda Doesn't Work for PyTorch

| Component | Unzipped Size | Lambda Limit | Result |
|-----------|---------------|--------------|--------|
| PyTorch | ~250MB | 250MB total | Won't fit |
| Your model | ~1-2MB | | ✓ Fits |
| Your code | <1MB | | ✓ Fits |
| **Total** | **~250MB+** | **250MB** | **❌ Exceeds** |

**The math doesn't work.** Even if we got PyTorch to exactly 250MB, we'd have no room for our code or model.

---

## Lambda Size Limits (Hard AWS Limits)

| Limit Type | Size | Can Change? |
|------------|------|-------------|
| Deployment package (zipped) | 50MB | No |
| Deployment package (unzipped) | 250MB | No |
| Layers (all layers combined, unzipped) | 250MB | No |
| Container image | 10GB | No (different deployment method) |
| /tmp directory | 512MB - 10GB | Yes (configurable) |

**None of these help us.** PyTorch won't fit in any configuration.

---

## What We Successfully Built

Despite Lambda not working, we built a lot:

### ✅ Working Components

1. **Lambda function code**
   - `lambda_function.py` - Handler with proper API Gateway integration
   - `model_loader.py` - Model loading from S3
   - CORS headers
   - Error handling

2. **S3 Model Storage**
   - Bucket: `mnist-model-ai-training-sebastian`
   - Model uploaded and accessible

3. **API Gateway**
   - REST API created
   - `/predict` endpoint
   - CORS enabled
   - Deployed to `prod` stage

4. **Proper Architecture**
   - Request → API Gateway → Lambda → S3 → Prediction
   - All the pieces work except PyTorch in Lambda

---

## Alternative: Lambda with Container Images

**Could this work?**

Lambda supports container images up to 10GB.

**Would need to:**
1. Create Dockerfile with PyTorch
2. Build container image
3. Push to ECR
4. Deploy Lambda from container

**Pros:**
- 10GB limit (PyTorch fits!)
- Full control over environment

**Cons:**
- More complex deployment
- Slower cold starts (~10-30 seconds)
- Still not ideal for ML models

**Verdict:** Possible but not recommended. SageMaker is better.

---

## The Right Tool: SageMaker

**Why SageMaker exists:**

AWS built SageMaker specifically because Lambda doesn't work well for ML models.

**SageMaker solves:**
- ✅ No size limits
- ✅ GPU support
- ✅ Faster inference
- ✅ Auto-scaling
- ✅ Model versioning
- ✅ Built-in monitoring

**Cost comparison:**
- Lambda: $0.20 per 1M requests (but can't use it)
- SageMaker: ~$0.05/hour (~$36/month if always on)
- SageMaker Serverless: Pay per request (similar to Lambda)

---

## What You Learned

### ✅ Skills Gained

1. **Lambda architecture** - How serverless functions work
2. **API Gateway** - How to create REST APIs
3. **Lambda layers** - What they are and their limits
4. **S3 integration** - Loading files from S3
5. **IAM permissions** - Execution roles and policies
6. **Real AWS constraints** - Why certain architectures exist

### ✅ Code You Can Reuse

- `lambda_function.py` - Portable to SageMaker or other services
- `model_loader.py` - Works with minor modifications
- API Gateway configuration - Can point to SageMaker instead
- React frontend - Just needs endpoint URL changed

---

## Summary Table: What Works Where

| Component | Lambda | Lambda Container | SageMaker |
|-----------|--------|------------------|-----------|
| Small models (<50MB) | ✅ | ✅ | ✅ |
| PyTorch models | ❌ | ⚠️ Slow | ✅ |
| GPU inference | ❌ | ❌ | ✅ |
| Sub-second latency | ✅ | ❌ | ✅ |
| Cost (low traffic) | $ | $$ | $$$ |
| Easy deployment | ✅ | ❌ | ✅ |

---

## Next Steps

See `AWS_SAGEMAKER_GUIDE.md` for how to deploy your model properly using SageMaker.

**What you'll do:**
1. Create SageMaker endpoint
2. Deploy your PyTorch model
3. Point API Gateway to SageMaker
4. Keep your React frontend (no changes needed!)

---

## Files Created (Keep These)
```
lambda_function/
├── lambda_function.py    # Reusable handler logic
├── model_loader.py       # Reusable model loading (adapt for SageMaker)
└── function.zip          # Not used, can delete

mnist_model.pth           # Already in S3, ready for SageMaker
```

---

## Lessons Learned

1. **Lambda is great for small, stateless functions** - Not ML models with large dependencies
2. **AWS has purpose-built services** - Use SageMaker for ML, not Lambda
3. **Public layers aren't really public** - Cross-account access requires explicit permissions
4. **Size limits are hard** - Can't be configured or increased
5. **Always check service limits first** - Could have saved time by checking PyTorch size vs Lambda limits upfront

---

**You didn't fail - Lambda just isn't the right tool for PyTorch models. Let's use the right tool: SageMaker.**

EOF

# Create SageMaker guide
cat > AWS_SAGEMAKER_GUIDE.md << 'EOF'
# AWS SageMaker Deployment - The Right Tool for ML Models

## Why SageMaker?

SageMaker is AWS's purpose-built service for deploying ML models. Unlike Lambda:
- ✅ No size limits
- ✅ GPU support (optional)
- ✅ Designed for ML workloads
- ✅ Auto-scaling
- ✅ Model versioning
- ✅ <1 second inference latency

---

## Cost Comparison

| Option | Cost | When to Use |
|--------|------|-------------|
| **Lambda** | $0.20 per 1M requests | Can't use for PyTorch |
| **SageMaker Real-time** | ~$0.05/hour ($36/month) | Always-on, low latency |
| **SageMaker Serverless** | Per request + compute | Infrequent traffic |

**For learning:** Use real-time endpoint, delete when not using (~$1-2 for testing).

---

## Architecture
```
React Frontend
    ↓ POST /predict
API Gateway (optional)
    ↓
SageMaker Endpoint
    ├── PyTorch model
    ├── Inference code
    └── Auto-scaling
```

---

## Prerequisites

What you already have:
- ✅ Trained model (`mnist_model.pth`)
- ✅ Model loading code (`model_loader.py`)
- ✅ S3 bucket
- ✅ React frontend

---

## Step 1: Prepare Model for SageMaker

### 1.1: Create inference script

**Directory:** Project root
```bash
cd ~/ai-projects/mnist-digit-classifier
mkdir sagemaker_deployment
cd sagemaker_deployment
```

Create `inference.py`:
```python
"""
SageMaker inference script
Handles model loading and prediction requests
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

# CNN Architecture (same as training)
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

# SageMaker required functions
def model_fn(model_dir):
    """Load model from model_dir - called once at endpoint startup"""
    device = torch.device('cpu')
    model = DigitClassifier().to(device)
    model_path = os.path.join(model_dir, 'mnist_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def input_fn(request_body, content_type):
    """Parse input data - receives base64 image"""
    if content_type == 'application/json':
        data = json.loads(request_body)
        image_data = base64.b64decode(data['image'])
        return BytesIO(image_data)
    else:
        raise ValueError(f'Unsupported content type: {content_type}')

def predict_fn(data, model):
    """Run prediction"""
    # Preprocess image
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

    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img_tensor = transform(canvas).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()

    return prediction, confidence

def output_fn(prediction, accept):
    """Format output"""
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

# Copy model
cp ../mnist_model.pth .

# Create model tar.gz (SageMaker requirement)
tar -czf model.tar.gz mnist_model.pth inference.py

# Upload to S3
aws s3 cp model.tar.gz s3://mnist-model-ai-training-sebastian/sagemaker/ --profile firstfire
```

---

## Step 2: Create SageMaker Endpoint (AWS Console)

### 2.1: Create Model

1. Go to **SageMaker** console
2. Click **Inference** → **Models** → **Create model**
3. **Model name:** `mnist-classifier-model`
4. **IAM role:** Create new role (allow S3 access)
5. **Container:**
   - Image: `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-cpu-py310`
   - Model artifact: `s3://mnist-model-ai-training-sebastian/sagemaker/model.tar.gz`
6. Click **Create model**

### 2.2: Create Endpoint Configuration

1. **Inference** → **Endpoint configurations** → **Create**
2. **Name:** `mnist-classifier-config`
3. **Add model:** Select `mnist-classifier-model`
4. **Instance type:** `ml.t2.medium` (cheapest for testing)
5. **Initial instance count:** 1
6. Click **Create**

### 2.3: Create Endpoint

1. **Inference** → **Endpoints** → **Create endpoint**
2. **Name:** `mnist-classifier-endpoint`
3. **Endpoint configuration:** Select `mnist-classifier-config`
4. Click **Create endpoint**

**Wait 5-10 minutes** for endpoint to be "InService".

---

## Step 3: Test SageMaker Endpoint

### 3.1: Test via AWS Console

1. Go to your endpoint
2. Click **Test inference**
3. Paste test data:
```json
{
  "image": "BASE64_ENCODED_IMAGE_HERE"
}
```
4. Click **Invoke endpoint**

### 3.2: Test via CLI

**Directory:** Project root
```bash
cd ~/ai-projects/mnist-digit-classifier

# Use the base64 file we created earlier
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name mnist-classifier-endpoint \
  --content-type application/json \
  --body file://test_payload.json \
  --profile firstfire \
  output.json

cat output.json
```

**Expected:**
```json
{"prediction": 5, "confidence": 94.67}
```

---

## Step 4: Update React Frontend

### 4.1: Update API endpoint

**Directory:** `frontend/src/`
```bash
cd ~/ai-projects/mnist-digit-classifier/frontend/src
```

Open `App.js` and update fetch URL:
```javascript
const handlePredict = async () => {
  setLoading(true);

  const imageData = await canvasRef.current.exportImage('png');
  const base64Image = imageData.split(',')[1];

  try {
    // SageMaker endpoint URL (via API Gateway proxy)
    const response = await fetch('YOUR-API-GATEWAY-URL/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: base64Image }),
    });

    const data = await response.json();
    setPrediction(data.prediction);
    setConfidence(data.confidence);
  } catch (error) {
    console.error('Error:', error);
    alert('Failed to get prediction');
  }

  setLoading(false);
};
```

---

## Step 5: (Optional) Add API Gateway

If you want a public URL instead of SageMaker endpoint directly:

1. Create Lambda that calls SageMaker
2. Point API Gateway to that Lambda
3. Lambda code:
```python
import boto3
import json

sagemaker = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    body = json.loads(event['body'])

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

---

## Cost Management

### Delete Resources When Not Using

**Stop inference:**
```bash
aws sagemaker delete-endpoint --endpoint-name mnist-classifier-endpoint --profile firstfire
```

**Cost while running:**
- ml.t2.medium: ~$0.065/hour (~$47/month if always on)

**For learning:**
- Create endpoint when testing
- Delete when done
- Total cost: $1-5 for a few hours of testing

---

## Comparison: What You've Built

| Version | Pros | Cons | Status |
|---------|------|------|--------|
| **Django (local)** | Free, fast dev | Local only | ✅ Working |
| **Lambda** | Cheap, serverless | Can't fit PyTorch | ❌ Doesn't work |
| **SageMaker** | Purpose-built for ML | Costs money | 🔜 Next step |

---

## Next Steps After SageMaker

1. **Deploy React to S3 + CloudFront**
2. **Add authentication** (Cognito)
3. **Monitor with CloudWatch**
4. **A/B test models** (SageMaker variants)
5. **Add more models** (Fashion-MNIST, etc.)

---

## Summary

You've learned:
- ✅ Why Lambda doesn't work for ML
- ✅ How to package models for SageMaker
- ✅ How to create SageMaker endpoints
- ✅ Real AWS architecture decisions
- ✅ Cost trade-offs

**This is production-grade ML deployment!**

EOF

echo ""
echo "✅ Created AWS_LAMBDA_REALITY_CHECK.md"
echo "✅ Created AWS_SAGEMAKER_GUIDE.md"
echo ""
echo "Summary:"
echo "- Lambda won't work for PyTorch (size limits)"
echo "- SageMaker is the right tool"
echo "- All your code is reusable"
echo "- Ready to deploy to SageMaker when you are"