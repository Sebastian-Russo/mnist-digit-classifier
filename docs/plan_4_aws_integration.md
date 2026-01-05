
cat > AWS_LAMBDA_PLAN.md << 'EOF'
# AWS Lambda + API Gateway - Action Plan

## Goal
Deploy your digit classifier as a serverless API using your existing AWS skills.

## What You'll Build
- Lambda function with PyTorch model
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
    ├── Load model from S3
    ├── Preprocess image
    ├── Run prediction
    └── Return JSON
```

---

## Steps

### 1. Prepare Lambda Package

**Create deployment folder:**
```bash
cd ~/ai-projects/mnist-digit-classifier
mkdir lambda_deployment
cd lambda_deployment
```

**Install dependencies:**
```bash
pip install torch torchvision pillow numpy -t .
```

**Copy your code:**
```bash
cp ../api/classifier/model_loader.py lambda_function.py
```

**Edit lambda_function.py to add handler:**
```python
def lambda_handler(event, context):
    # Parse image from event
    # Call predict()
    # Return response
```

### 2. Upload Model to S3

**Create S3 bucket:**
```bash
aws s3 mb s3://mnist-model-YOUR-NAME
```

**Upload model:**
```bash
aws s3 cp ../mnist_model.pth s3://mnist-model-YOUR-NAME/
```

### 3. Update Lambda Code

Modify `lambda_function.py` to:
- Load model from S3 (not local file)
- Handle base64 encoded images
- Return API Gateway compatible response

### 4. Create Lambda Function

**Via AWS Console:**
1. Create function: `mnist-digit-predictor`
2. Runtime: Python 3.12
3. Architecture: x86_64
4. Memory: 512 MB
5. Timeout: 30 seconds

**Add environment variables:**
```
MODEL_BUCKET=mnist-model-YOUR-NAME
MODEL_KEY=mnist_model.pth
```

### 5. Package and Deploy

**Zip the package:**
```bash
cd ~/ai-projects/mnist-digit-classifier/lambda_deployment
zip -r lambda_package.zip .
```

**Upload to Lambda:**
```bash
aws lambda update-function-code \
  --function-name mnist-digit-predictor \
  --zip-file fileb://lambda_package.zip
```

### 6. Add S3 Permissions

**Attach policy to Lambda role:**
- AmazonS3ReadOnlyAccess

Or create custom policy:
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject"],
  "Resource": "arn:aws:s3:::mnist-model-YOUR-NAME/*"
}
```

### 7. Create API Gateway

**REST API:**
1. Create API: `mnist-api`
2. Create resource: `/predict`
3. Create method: `POST`
4. Integration type: Lambda Function
5. Select: `mnist-digit-predictor`

**Enable CORS:**
1. Actions → Enable CORS
2. Deploy API to stage: `prod`

### 8. Test

**Get invoke URL from API Gateway**

**Test with curl:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_ENCODED_IMAGE"}' \
  https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/predict
```

### 9. Update React Frontend

Change API endpoint from:
```javascript
http://localhost:8000/predict/
```

To:
```javascript
https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/predict
```

---

## Challenges You'll Face

### 1. Lambda Package Size
- PyTorch is ~100MB
- Lambda limit: 250MB
- **Solution:** Use Lambda Layers or slim PyTorch build

### 2. Cold Starts
- First request takes 3-5 seconds
- **Solution:** Keep Lambda warm with CloudWatch Events

### 3. Image Encoding
- Need to convert image to base64
- **Solution:** Handle in API Gateway or client

---

## Estimated Time: 2-3 hours

## Cost
- Lambda: ~$0.20 per 1M requests
- API Gateway: ~$3.50 per 1M requests
- S3: ~$0.023 per GB/month
- **Total for learning:** Essentially free (<$1/month)

---

## Alternative: Use SageMaker

**When Lambda doesn't work:**
- Model too large (>250MB)
- Need GPU inference
- High-traffic production

**SageMaker Endpoint:**
- Managed infrastructure
- Auto-scaling
- Pay per hour ($0.05-$0.50/hr)
EOF

echo "✅ Created REACT_FRONTEND_PLAN.md"
echo "✅ Created AWS_LAMBDA_PLAN.md"
