from django.urls import path
from . import views

app_name = 'rag'

urlpatterns = [
    path('healthz/', views.health_check, name='health_check'),
    path('reindex/', views.ReindexView.as_view(), name='reindex'),
    path('ask/', views.AskView.as_view(), name='ask'),
    path('generate-ppt/', views.GeneratePPTView.as_view(), name='generate_ppt'),
]
