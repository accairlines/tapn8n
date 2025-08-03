"""
WSGI config for aet_predictor project.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aet_predictor.settings')

application = get_wsgi_application() 