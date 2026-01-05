"""
ANALOGY: The restaurant's menu
Maps customer requests to the right waiter (view function)
"""

from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_digit, name='predict_digit'),
]
