from django.urls import re_path
from . import views

app_name = 'rag'

urlpatterns = [
    re_path(r'^healthz$', views.health_check, name='health_check'),
    re_path(r'^reindex$', views.reindex, name='reindex'),
    re_path(r'^ask$', views.ask, name='ask'),
    re_path(r'^generate-ppt$', views.generate_ppt, name='generate_ppt'),
    re_path(r'^sync-emails$', views.sync_emails, name='sync_emails'),
]
