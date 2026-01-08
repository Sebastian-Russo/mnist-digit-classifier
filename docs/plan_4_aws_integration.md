cd ~/ai-projects/mnist-digit-classifier
cat > AWS_LAMBDA_PLAN.md << 'EOF'
# AWS Lambda + API Gateway - Action Plan

## Goal
Deploy your digit classifier as a serverless API using your existing AWS skills.

## What You'll Build
- Lambda function with PyTorch model
- Lambda Layer for PyTorch (reusable across functions)
- API Gateway endpoint
- S3 bucket for model storage
- Public API accessible from anywhere

---

## Architecture
```
Client (Postman/React)
    ↓ POST /predict
API Gateway
    ↓ Trigger
Lambda Function
    ├── Uses Lambda Layer (PyTorch)
    ├── Load model from S3
    ├── Preprocess image
    ├── Run prediction
    └── Return JSON
```

---

## Steps

### 1. Create Lambda Layer (PyTorch Dependencies)

**Why:** Lambda has 250MB limit. PyTorch alone is ~100MB. Layers let you reuse dependencies across functions.

**Create layer folder:**
```bash
cd ~/ai-projects/mnist-digit-classifier
mkdir -p lambda_layer/python
cd lambda_layer/python
```

**Install dependencies to layer:**
```bash
pip install torch torchvision pillow numpy -t .
```

**Package breakdown:**
- **torch** (~100MB) - PyTorch ML framework, runs your CNN model
- **torchvision** (~10MB) - Image preprocessing utilities (transforms, etc.)
- **pillow** (~5MB) - Image loading/manipulation (PIL)
- **numpy** (~20MB) - Array operations (required by PyTorch)

**Zip the layer:**
```bash
cd ~/ai-projects/mnist-digit-classifier/lambda_layer
zip -r pytorch_layer.zip python/
```

**Create layer in AWS:**
```bash
aws lambda publish-layer-version \
  --layer-name pytorch-layer \
  --zip-file fileb://pytorch_layer.zip \
  --compatible-runtimes python3.12
```

**Note the Layer ARN** - you'll need it later!

---

### 2. Prepare Lambda Function Code

**Create function folder:**
```bash
cd ~/ai-projects/mnist-digit-classifier
mkdir lambda_function
cd lambda_function
```

**Copy your model loader:**
```bash
cp ../api/classifier/model_loader.py .
```

**Create lambda_function.py:**
```python
import json
import base64
import boto3
import os
from io import BytesIO
from model_loader import predict

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda handler - receives image, returns prediction
    Same as Django views.py but for Lambda
    """
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))

        # Get base64 image
        if 'image' not in body:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'No image provided'})
            }

        # Decode base64 image
        image_data = base64.b64decode(body['image'])
        image_file = BytesIO(image_data)

        # Make prediction
        prediction, confidence = predict(image_file)

        # Return result
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  # CORS
            },
            'body': json.dumps({
                'prediction': int(prediction),
                'confidence': round(float(confidence) * 100, 2)
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
```

**Update model_loader.py to load from S3:**

Replace the model loading section with:
```python
import boto3

# Load model from S3
s3 = boto3.client('s3')
device = torch.device('cpu')  # Lambda doesn't have GPU
model = DigitClassifier().to(device)

# Download model from S3
model_bucket = os.environ.get('MODEL_BUCKET')
model_key = os.environ.get('MODEL_KEY', 'mnist_model.pth')

s3.download_file(model_bucket, model_key, '/tmp/mnist_model.pth')
model.load_state_dict(torch.load('/tmp/mnist_model.pth', map_location=device))
model.eval()
```

**Package function code:**
```bash
cd ~/ai-projects/mnist-digit-classifier/lambda_function
zip -r function.zip lambda_function.py model_loader.py
```

---

### 3. Upload Model to S3

**Create S3 bucket:**
```bash
aws s3 mb s3://mnist-model-YOUR-NAME
```

**Upload model:**
```bash
aws s3 cp ~/ai-projects/mnist-digit-classifier/mnist_model.pth s3://mnist-model-YOUR-NAME/
```

---

### 4. Create Lambda Function

**Via AWS Console:**
1. Go to Lambda console
2. Create function: `mnist-digit-predictor`
3. Runtime: Python 3.12
4. Architecture: x86_64
5. Create function

**Upload function code:**
```bash
aws lambda update-function-code \
  --function-name mnist-digit-predictor \
  --zip-file fileb://function.zip
```

**Attach PyTorch Layer:**
```bash
aws lambda update-function-configuration \
  --function-name mnist-digit-predictor \
  --layers YOUR-LAYER-ARN
```

**Configure function:**
- Memory: 512 MB
- Timeout: 30 seconds
- Environment variables:
  - `MODEL_BUCKET=mnist-model-YOUR-NAME`
  - `MODEL_KEY=mnist_model.pth`

---

### 5. Add S3 Permissions

**Via AWS Console:**
1. Go to Lambda function → Configuration → Permissions
2. Click on the execution role
3. Add permissions → Attach policies
4. Add: `AmazonS3ReadOnlyAccess`

**Or via CLI:**
```bash
aws iam attach-role-policy \
  --role-name YOUR-LAMBDA-ROLE \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

---

### 6. Create API Gateway

**Via AWS Console:**

1. **Create REST API:**
   - API Gateway console → Create API
   - REST API (not HTTP API)
   - Name: `mnist-api`

2. **Create Resource:**
   - Actions → Create Resource
   - Resource Name: `predict`
   - Resource Path: `/predict`

3. **Create Method:**
   - Select `/predict` resource
   - Actions → Create Method → POST
   - Integration type: Lambda Function
   - Lambda Function: `mnist-digit-predictor`
   - Save

4. **Enable CORS:**
   - Select `/predict` resource
   - Actions → Enable CORS
   - Use default settings
   - Enable CORS and replace existing

5. **Deploy API:**
   - Actions → Deploy API
   - Deployment stage: `prod`
   - Deploy

6. **Get Invoke URL:**
   - Copy the invoke URL (looks like: `https://abc123.execute-api.us-east-1.amazonaws.com/prod`)

---

### 7. Test with curl

**Convert image to base64:**
```bash
base64 ~/ai-projects/mnist-digit-classifier/test_images/digit_5.png > digit5_base64.txt
```

**Remove newlines:**
```bash
tr -d '\n' < digit5_base64.txt > digit5_clean.txt
```

**Create test JSON:**
```bash
echo "{\"image\": \"$(cat digit5_clean.txt)\"}" > test_payload.json
```

**Test the API:**
```bash
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

### 8. Update React Frontend

**File:** `frontend/src/App.js`

Update the fetch URL and change image encoding:
```javascript
const handlePredict = async () => {
  setLoading(true);

  // Export canvas as base64
  const imageData = await canvasRef.current.exportImage('png');

  // Remove data URL prefix
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

**Test your app** - should now use Lambda instead of Django!

---

## Challenges & Solutions

### 1. Lambda Package Size (SOLVED with Layers)
- ✅ PyTorch in reusable layer
- ✅ Function code stays small

### 2. Cold Starts (~3-5 seconds first request)
**Solutions:**
- Provision concurrency (keeps Lambda warm)
- CloudWatch Events to ping every 5 minutes
- Accept it for low-traffic apps

### 3. Base64 Encoding
- Images must be base64 encoded
- Handled in React frontend

---

## Estimated Time: 2-3 hours

## Cost Breakdown
- **Lambda:** ~$0.20 per 1M requests
- **API Gateway:** ~$3.50 per 1M requests
- **S3:** ~$0.023 per GB/month
- **Lambda Layer:** Free (just storage)
- **Total for learning:** <$1/month

---

## When to Use SageMaker Instead

**Use Lambda when:**
- ✅ Small models (<250MB with layers)
- ✅ Infrequent requests (cold starts OK)
- ✅ Want serverless simplicity

**Use SageMaker when:**
- ❌ Model >250MB (even with layers)
- ❌ Need GPU inference
- ❌ High traffic (>100 req/min)
- ❌ Need <1s response time consistently

---

## Next Steps After Lambda

1. Add CloudWatch monitoring
2. Set up CloudWatch alarms
3. Implement caching (API Gateway cache)
4. Add authentication (API keys or Cognito)
5. Deploy React to S3 + CloudFront

EOF

echo "✅ Updated AWS_LAMBDA_PLAN.md with Lambda Layers approach"