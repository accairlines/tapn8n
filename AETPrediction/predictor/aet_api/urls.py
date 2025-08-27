from django.urls import path
from . import views

urlpatterns = [
    path('', views.api_info, name='api_info'),
    path('health/', views.health_check, name='health_check'),
    path('predict/<int:flight_id>/', views.predict_flight, name='predict_flight'),
    path('predict/batch/', views.predict_batch, name='predict_batch'),
] 