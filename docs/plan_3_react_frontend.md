cd ~/ai-projects/mnist-digit-classifier
cat > REACT_FRONTEND_PLAN.md << 'EOF'
# React Frontend - Action Plan

## Goal
Build a web app where users can draw digits and get instant predictions from your Django API.

## What You'll Build
- Canvas to draw digits
- "Predict" button
- Display prediction + confidence
- "Clear" button to reset

## Prerequisites
```bash
# Install Node.js (if not installed)
node --version  # Should show v18+
npm --version   # Should show 9+
```

---

## Steps

### 1. Create React App
```bash
cd ~/ai-projects/mnist-digit-classifier
npx create-react-app frontend
cd frontend
```

### 2. Install Dependencies
```bash
npm install react-canvas-draw // version conflict

npm install react-sketch-canvas
```

### 3. Update CORS in Django
**File:** `api/api/settings.py`

Install django-cors-headers:
```bash
cd ~/ai-projects/mnist-digit-classifier/api
source ../venv/bin/activate
pip install django-cors-headers
```

Add to `INSTALLED_APPS`:
```python
INSTALLED_APPS = [
    ...
    'corsheaders',
]
```

Add to `MIDDLEWARE` (at the top):
```python
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    ...
]
```

Add to bottom of settings.py:
```python
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
]
```

### 4. Create Drawing Component
**File:** `frontend/src/App.js`

Replace with canvas + prediction logic

### 5. Add Styling
**File:** `frontend/src/App.css`

Make it look nice

### 6. Test Locally
**Terminal 1 (Django):**
```bash
cd ~/ai-projects/mnist-digit-classifier/api
source ../venv/bin/activate
python3 manage.py runserver
```

**Terminal 2 (React):**
```bash
cd ~/ai-projects/mnist-digit-classifier/frontend
npm start
```

Visit: http://localhost:3000

### 7. Draw & Predict
- Draw a digit
- Click "Predict"
- See result!

---

## Estimated Time: 30-45 minutes

## Final Result
Web app that works like:
1. Draw digit on canvas
2. Click "Predict"
3. See: "Prediction: 7 (Confidence: 89.3%)"
4. Click "Clear" to try again
EOF
