from django.conf import settings
from rest_framework import authentication
from rest_framework import exceptions
from rest_framework.authentication import TokenAuthentication


class APIKeyAuthentication(authentication.BaseAuthentication):
    """
    Custom authentication class that validates requests using an API key.
    The API key should be provided in the X-API-Key header.
    """
    
    def authenticate(self, request):
        # Get the API key from the request header
        api_key = request.META.get('HTTP_X_API_KEY')
        
        if not api_key:
            return None
        
        # Check if the provided API key matches the configured one
        if api_key == settings.API_AUTH_TOKEN:
            # Return a tuple of (user, auth) - user can be None for API key auth
            return (None, api_key)
        
        return None
    
    def authenticate_header(self, request):
        return 'X-API-Key'


class FallbackAuthentication(authentication.BaseAuthentication):
    """
    Fallback authentication that tries API key first, then token authentication.
    This allows for backward compatibility if you want to support both methods.
    """
    
    def __init__(self):
        self.api_key_auth = APIKeyAuthentication()
        self.token_auth = TokenAuthentication()
    
    def authenticate(self, request):
        # Try API key authentication first
        user_auth = self.api_key_auth.authenticate(request)
        if user_auth:
            return user_auth
        
        # Fall back to token authentication
        return self.token_auth.authenticate(request)
    
    def authenticate_header(self, request):
        return 'X-API-Key or Authorization'
