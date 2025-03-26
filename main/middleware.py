from django.shortcuts import redirect
from django.contrib import messages
from django.urls import resolve
from db_connection import db

users_collection = db['users']

class AuthenticationMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Get the current URL name
        url_name = resolve(request.path_info).url_name if hasattr(resolve(request.path_info), 'url_name') else None
        
        # Define protected URLs
        admin_protected_urls = ['admin_dashboard']
        user_protected_urls = ['user_dashboard']
        all_protected_urls = ['profile'] + admin_protected_urls + user_protected_urls
        
        # Check if the URL requires authentication
        if url_name in all_protected_urls:
            # Check if user is logged in
            pb_number = request.session.get('pb_number')
            if not pb_number:
                messages.error(request, 'Please login to access this page')
                return redirect('signin')
            
            # Verify user in database
            user_data = users_collection.find_one({"pb_number": pb_number})
            if not user_data:
                # Invalid user, clear session
                if 'pb_number' in request.session:
                    del request.session['pb_number']
                if 'role' in request.session:
                    del request.session['role']
                messages.error(request, 'Invalid session. Please login again.')
                return redirect('signin')
            
            # Check admin-specific permissions
            if url_name in admin_protected_urls and user_data.get('role', '').lower() != 'admin':
                messages.error(request, 'Access denied. Admin access only.')
                return redirect('signin')
            
            # Check user-specific permissions
            if url_name in user_protected_urls and user_data.get('role', '').lower() != 'user':
                messages.error(request, 'Access denied. User access only.')
                return redirect('signin')
        
        response = self.get_response(request)
        return response