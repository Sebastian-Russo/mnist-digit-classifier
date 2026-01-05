cat > DJANGO_API_PLAN.md << 'EOF'
# Django API Build Plan

## Goal
Build an API endpoint that accepts digit images and returns predictions.

## Steps

### 1. Setup (✅ DONE)
- [x] Install Django
- [x] Create project: `django-admin startproject api`
- [x] Create app: `python manage.py startapp classifier`
- [x] Add `'classifier'` to `INSTALLED_APPS` in `api/settings.py`
- [x] Move SECRET_KEY to `.env` file
- [x] Test server runs: `python manage.py runserver`

### 2. Copy Model File
```bash
cp ~/ai-projects/mnist-digit-classifier/mnist_model.pth ~/ai-projects/mnist-digit-classifier/api/classifier/
```

### 3. Create Model Loader (`classifier/model_loader.py`)
- Define CNN architecture (same as training)
- Load trained weights
- Provide predict function

### 4. Create API Endpoint (`classifier/views.py`)
- Accept POST request with image
- Preprocess image (same as predict.py)
- Run prediction
- Return JSON response

### 5. Create URL Route (`classifier/urls.py` + `api/urls.py`)
- Map `/predict/` to the view function

### 6. Test the API
```bash
# Test with curl
curl -X POST -F "image=@../test_images/digit_5.png" http://localhost:8000/predict/
```

Expected response:
```json
{"prediction": 5, "confidence": 0.95}
```

## Estimated Time: 20-30 minutes
EOF