🍕 Django = A Pizza Restaurant
Django Concept              Pizza Restaurant
Django Project (api/)       The whole restaurant
Django App (classifier/)    The kitchen (where food is made)
settings.py                 Restaurant policies & approved kitchens
urls.py                     The menu (what customers can order)
views.py                    The chefs (handle orders, make food)
manage.py                   The manager (runs everything)


**File**                    **Purpose**                         **Restaurant Analogy**
model_loader.py             Business logic (your model)         The kitchen & chef
views.py                    Handle requests, return responses   The waiter
urls.py                     Map URLs to views                   The menu

models.py                   Database schemas                    (Empty - you don't store data)
admin.py                    Admin interface                     (Empty - no admin needed)
tests.py                    Unit tests                          (Empty - no tests yet)


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



cd ~/ai-projects/mnist-digit-classifier
cat > DJANGO_VS_FRAMEWORKS.md << 'EOF'
# Django vs Familiar Frameworks

Understanding Django through the lens of Express.js and AWS Lambda

---

## Django vs Express.js

| Concept | Django | Express (Node.js) |
|---------|--------|-------------------|
| **Project setup** | `django-admin startproject` | `npm init` + `express()` |
| **Routing** | `urls.py` | `app.get()` / `app.post()` |
| **Request handlers** | `views.py` functions | Route handler functions |
| **Middleware** | Django middleware | `app.use()` |
| **Start server** | `manage.py runserver` | `app.listen(3000)` |

---

## Code Comparison: Django vs Express

### Django (what you built):
```python
# urls.py (routing)
urlpatterns = [
    path('predict/', views.predict_digit),
]

# views.py (handler)
def predict_digit(request):
    image = request.FILES['image']
    prediction, confidence = predict(image)
    return JsonResponse({'prediction': prediction})
```

### Express equivalent:
```javascript
// Routing + handler together
app.post('/predict', (req, res) => {
    const image = req.files.image;
    const {prediction, confidence} = predict(image);
    res.json({prediction: prediction});
});
```

**Same flow, different syntax!**

---

## Django vs AWS API Gateway + Lambda

| Django | AWS Serverless |
|--------|----------------|
| **urls.py** | API Gateway routes |
| **views.py** | Lambda function handler |
| **manage.py runserver** | API Gateway deployment |
| **Request object** | Lambda event object |
| **JsonResponse** | Lambda return value |

---

## Code Comparison: Django vs Lambda

### Django:
```python
# views.py
def predict_digit(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    image = request.FILES['image']
    prediction, confidence = predict(image)

    return JsonResponse({
        'prediction': prediction,
        'confidence': confidence
    })
```

### Lambda equivalent:
```python
# lambda_function.py
def lambda_handler(event, context):
    if event['httpMethod'] != 'POST':
        return {
            'statusCode': 405,
            'body': json.dumps({'error': 'POST only'})
        }

    image = event['body']  # base64 encoded
    prediction, confidence = predict(image)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'prediction': prediction,
            'confidence': confidence
        })
    }
```

**Same logic, different wrapper!**

---

## Amazon Connect Context

In your Connect flows:

| Connect Concept | Django Equivalent |
|-----------------|-------------------|
| **Contact flow block** | URL route |
| **Lambda function invocation** | View function call |
| **Lambda response** | JsonResponse |
| **Contact attributes** | Request data |

You're already doing this! Just different names.

---

## The Universal Pattern

All web frameworks follow the same pattern:
```
1. Client sends request
   ↓
2. Router decides which handler
   ↓
3. Handler processes request
   ↓
4. Business logic executes
   ↓
5. Handler returns response
```

**Django, Express, Lambda - all same pattern, different syntax!**

---

## Key Takeaway

If you understand AWS Lambda + API Gateway, you already understand Django. The concepts are identical, just packaged differently.
