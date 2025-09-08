from django.urls import path
from . import views

urlpatterns = [
    path('predict/<int:flight_id>/', views.predict_flight, name='predict_flight'),
    path('predict/batch/', views.predict_batch, name='predict_batch')
] 