"""
ANALOGY: Main restaurant entrance
Directs customers to the right section of the menu
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('classifier.urls')),  # Routes /predict/ to classifier
]
