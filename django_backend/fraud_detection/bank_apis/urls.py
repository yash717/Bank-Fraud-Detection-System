from django.urls import path
from . import views

urlpatterns = [
    path('predict_fraud/', views.predict_fraud_view, name='predict_fraud'),
]
