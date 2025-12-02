from django.urls import path
from . import views

urlpatterns = [
    path('predict/<int:flight_id>/', views.predict_flight, name='predict_flight'),
    path('train_model/', views.train_model, name='train_model')
] 