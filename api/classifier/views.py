"""
ANALOGY: The waiter who takes orders from customers
Customer sends image → Waiter receives it → Sends to kitchen (model_loader)
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .model_loader import predict

@csrf_exempt  # Allow external requests (for testing)
def predict_digit(request):
    """
    ANALOGY: Waiter takes customer's order
    Customer orders: "What digit is this?" (sends image)
    Waiter forwards to kitchen (model_loader.predict)
    Returns result to customer
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

    # Check if customer sent an image
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image provided'}, status=400)

    # Get the image
    image_file = request.FILES['image']

    # Send to kitchen (detective agency)
    prediction, confidence = predict(image_file)

    # Serve result to customer
    return JsonResponse({
        'prediction': prediction,
        'confidence': round(confidence * 100, 2)
    })

###############################################
# Historical naming from MVC pattern:
# MVC = Model-View-Controller (traditional web framework pattern)
# MVC TermDjango TermWhat It DoesModelmodels.pyData/databaseViewtemplates/ (HTML)What user seesControllerviews.pyBusiness logic

# Why It's Confusing
# Django switched terminology:

# "View" in Django = Controller logic (handles requests)
# "Template" in Django = What other frameworks call "View"

# So views.py is not what the user sees - it's the logic that decides what to show.

# Better Name Would Be
# controllers.py or handlers.py - but Django chose views.py and everyone stuck with it.
# Restaurant analogy: Should be called waiters.py but historically named views.py.

# TL;DR: Historical naming quirk. Just remember: views.py = request handlers.
