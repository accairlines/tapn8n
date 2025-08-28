from rest_framework import permissions


class HasValidAPIKey(permissions.BasePermission):
    """
    Permission class that checks if the request has a valid API key.
    This works with the APIKeyAuthentication class.
    """
    
    def has_permission(self, request, view):
        # If the request has been authenticated (has a valid API key), allow access
        return request.auth is not None

