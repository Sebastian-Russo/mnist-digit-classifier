# SageMaker Deployment - Debugging Summary

## What We Built
Successfully deployed MNIST digit classifier to AWS SageMaker endpoint that accepts images via API and returns predictions.

---

## Issues We Debugged

### Issue 1: Lambda Size Limits
**Problem:** Tried to deploy PyTorch model to Lambda
**Error:** PyTorch (250MB+ unzipped) exceeds Lambda's 250MB limit
**Solution:** Switched to SageMaker (no size limits)
**Learning:** Lambda is for small functions, SageMaker is for ML models

---

### Issue 2: Model Caching
**Problem:** Updated inference.py code but endpoint kept using old version
**Error:** Same errors repeated even after uploading new code
**Solution:** Had to create NEW model + NEW endpoint each time, not just update
**Learning:** SageMaker caches model artifacts when endpoint starts - can't hot-reload

---

### Issue 3: BytesIO Handling (Main Bug)
**Problem:** `input_fn` returned BytesIO, then `predict_fn` tried to open it again
**Error:** `PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object>`

**What we tried (all failed):**
- Adding `data.seek(0)` to reset BytesIO position
- Different S3 paths
- Recreating endpoints multiple times
- Checking logs (helpful but didn't point to root cause)

**Root cause:** Wrong separation of concerns
```python
# WRONG (what we were doing):
def input_fn():
    return BytesIO(image_data)  # Return raw bytes

def predict_fn(data, model):
    data.seek(0)
    img = Image.open(data)  # Try to open and preprocess
    # ...preprocess here
```

**Solution:** Do ALL preprocessing in `input_fn`, return ready-to-use tensor
```python
# CORRECT:
def input_fn():
    img = Image.open(BytesIO(image_data))
    # ...ALL preprocessing here
    tensor = transform(img)
    return tensor  # Return preprocessed tensor

def predict_fn(data, model):
    output = model(data)  # data is already tensor
    return prediction
```

**Learning:** SageMaker expects `input_fn` to return preprocessed data ready for the model, not raw bytes. Found solution by searching AWS docs showing proper inference.py structure.

---

### Issue 4: Invalid Test Data
**Problem:** test_payload.json had corrupted/invalid base64 image data
**Error:** Same PIL error as Issue 3
**Solution:** Created new test file with valid PNG image converted to base64
**Learning:** Always test image data locally before sending to endpoint

---

## Final Working Architecture
```
Jupyter Notebook
    ↓ POST with base64 image
SageMaker Endpoint (mnist-endpoint-sebastian-v4)
    ├── Model: mnist-model-sebastian-v4
    ├── Config: mnist-config-v4
    ├── Instance: ml.t2.medium
    └── Cost: ~$0.065/hour
    ↓
Response: {"prediction": 0, "confidence": 99.39}
```

---

## Key Files

| File | Location | Purpose |
|------|----------|---------|
| `inference.py` | S3: `model-final.tar.gz` | Preprocessing + model inference |
| `mnist_model.pth` | S3: `model-final.tar.gz` | Trained model weights |
| Model artifact | `s3://mnist-model-ai-training-sebastian/sagemaker/model-final.tar.gz` | Complete package |

---

## Cost Management

### Delete Endpoint (Stop Charges)

**In Jupyter notebook:**
```python
import boto3

sagemaker = boto3.client('sagemaker')

# Delete endpoint
sagemaker.delete_endpoint(EndpointName='mnist-endpoint-sebastian-v4')

print("✓ Deleted - no more charges")
```

**Cost while running:** ~$0.065/hour (~$47/month if left on)
**Cost while deleted:** $0

---

## Restart Endpoint Next Session

### Option 1: Via Jupyter Notebook (Recommended)
```python
import boto3

sagemaker = boto3.client('sagemaker')

# Check if endpoint exists
try:
    response = sagemaker.describe_endpoint(EndpointName='mnist-endpoint-sebastian-v4')
    print(f"Endpoint status: {response['EndpointStatus']}")
except:
    print("Endpoint doesn't exist - creating...")

    # Recreate endpoint (model + config already exist)
    sagemaker.create_endpoint(
        EndpointName='mnist-endpoint-sebastian-v4',
        EndpointConfigName='mnist-config-v4'
    )
    print("✓ Creating endpoint - wait 5-10 min for InService")

# Wait for InService
import time
while True:
    response = sagemaker.describe_endpoint(EndpointName='mnist-endpoint-sebastian-v4')
    status = response['EndpointStatus']
    print(f"Status: {status}")
    if status == 'InService':
        print("✓ Ready!")
        break
    elif status == 'Failed':
        print("✗ Failed to create")
        break
    time.sleep(30)
```

### Option 2: Via AWS Console

1. Go to **SageMaker** → **Endpoints**
2. Click **Create endpoint**
3. **Endpoint name:** `mnist-endpoint-sebastian-v4`
4. **Attach endpoint configuration:** Select existing config `mnist-config-v4`
5. Click **Create endpoint**
6. Wait 5-10 minutes for "InService"

---

## Test the Endpoint
```python
import boto3
import json

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Load test data (must be valid base64 PNG image)
with open('test_payload_fixed.json', 'r') as f:
    test_data = json.load(f)

# Invoke endpoint
response = sagemaker_runtime.invoke_endpoint(
    EndpointName='mnist-endpoint-sebastian-v4',
    ContentType='application/json',
    Body=json.dumps(test_data)
)

# Parse result
result = json.loads(response['Body'].read())
print(result)
# Expected: {"prediction": 0, "confidence": 99.39}
```

---

## Troubleshooting

### If endpoint fails to create:
1. Check CloudWatch logs: `/aws/sagemaker/Endpoints/mnist-endpoint-sebastian-v4`
2. Verify model exists: `sagemaker.describe_model(ModelName='mnist-model-sebastian-v4')`
3. Verify config exists: `sagemaker.describe_endpoint_config(EndpointConfigName='mnist-config-v4')`

### If getting 500 errors:
1. Check CloudWatch logs for actual Python error
2. Verify image data is valid base64 PNG
3. Test image decode locally first

### If model not found:
- Model and config are saved permanently in your account
- They persist even after deleting endpoints
- Only need to recreate endpoint, not model/config

---

## What You Learned

1. **Lambda limits** - Not suitable for large ML models (250MB limit)
2. **SageMaker architecture** - Model → Config → Endpoint separation
3. **Model caching** - Can't hot-reload, need new model/endpoint for code changes
4. **SageMaker inference pattern** - Preprocessing in `input_fn`, inference in `predict_fn`
5. **Cost management** - Delete endpoints when not using (~$0.065/hour)
6. **Debugging ML deployments** - Check logs, test locally, verify artifacts in S3

---

## Next Steps

- **Connect React frontend** to SageMaker endpoint
- **Add API Gateway** for public URL
- **Deploy React to S3** + CloudFront
- **Try different models** (Fashion-MNIST, etc.)
- **Add authentication** (Cognito)

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `sagemaker.create_endpoint()` | Start endpoint |
| `sagemaker.delete_endpoint()` | Stop endpoint (stop charges) |
| `sagemaker.describe_endpoint()` | Check status |
| `sagemaker_runtime.invoke_endpoint()` | Make prediction |

**Remember:** Model + Config are permanent. Endpoint is temporary (create/delete as needed).
