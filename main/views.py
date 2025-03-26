from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password, check_password
from .forms import LoginForm, UserRegistrationForm
from .models import User, users_collection
from pymongo import MongoClient

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['eguru']
users_collection = db['users']

def signin(request):
    if request.method == 'POST':
        pb_number = request.POST.get('pb_number')
        password = request.POST.get('password')
        user_data = users_collection.find_one({"pb_number": pb_number})

        if user_data and check_password(password, user_data.get('password')):
            request.session['pb_number'] = pb_number
            request.session['role'] = user_data.get('role').lower()

            print(f"Session set: pb_number={pb_number}, role={user_data.get('role').lower()}")

            if user_data.get('role').lower() == 'admin':
                return redirect('admin_dashboard')
            else:
                return redirect('user_dashboard')
        else:
            messages.error(request, 'Invalid PB Number or Password')

    return render(request, 'main/signin.html')

def signup(request):
    if request.method == 'POST':
        pb_number = request.POST.get('pb_number')
        password = request.POST.get('password')
        name = request.POST.get('name')
        gender = request.POST.get('gender')
        role = request.POST.get('role')
        division = request.POST.get('division')
        department = request.POST.get('department')
        designation = request.POST.get('designation')

        if users_collection.find_one({"pb_number": pb_number}):
            messages.error(request, 'PB Number already exists.')
        else:
            hashed_password = make_password(password)
            user_data = {
                "pb_number": pb_number,
                "password": hashed_password,
                "name": name,
                "gender": gender,
                "role": role,
                "division": division,
                "department": department,
                "designation": designation
            }
            result = users_collection.insert_one(user_data)
            print(f"User Inserted with ID: {result.inserted_id}")
            messages.success(request, 'Account created successfully. Please sign in.')
            return redirect('signin')

    return render(request, 'main/signup.html')


def signout(request):
    # Clear specific session variables first
    if 'pb_number' in request.session:
        del request.session['pb_number']
    
    if 'role' in request.session:
        del request.session['role']
    
    # Then clear all Django authentication
    logout(request)
    
    # Flush the entire session to ensure everything is cleared
    request.session.flush()
    
    # Optional: Reset the session cookie
    request.session.clear_expired()
    
    # Debug output to confirm session clearing
    print("Session cleared during logout")
    
    messages.success(request, 'You have been logged out.')
    return redirect('signin')

def home(request):
    return render(request, 'main/home.html')

# Admin Dashboard with session validation

def admin_dashboard(request):
    # The middleware already verified that this is a valid admin user
    pb_number = request.session.get('pb_number')
    user_data = users_collection.find_one({"pb_number": pb_number})
    all_users = list(users_collection.find({}))
    return render(request, 'main/admin_dashboard.html', {'all_users': all_users, 'admin': user_data})

def user_dashboard(request):
    pb_number = request.session.get('pb_number')
    user_data = users_collection.find_one({"pb_number": pb_number})
    return render(request, 'main/user_dashboard.html', {'user': user_data})

def create_user(request):
    if request.method == 'POST':
        pb_number = request.POST.get('pb_number')
        password = request.POST.get('password')
        name = request.POST.get('name')
        gender = request.POST.get('gender')
        role = request.POST.get('role')
        division = request.POST.get('division')
        department = request.POST.get('department')
        designation = request.POST.get('designation')

        if users_collection.find_one({"pb_number": pb_number}):
            messages.error(request, 'PB Number already exists.')
        else:
            hashed_password = make_password(password)
            user_data = {
                "pb_number": pb_number,
                "password": hashed_password,
                "name": name,
                "gender": gender,
                "role": role,
                "division": division,
                "department": department,
                "designation": designation
            }
            users_collection.insert_one(user_data)
            messages.success(request, 'User created successfully.')        
        return redirect('admin_dashboard')
    return render(request, 'main/admin_dashboard.html')

def profile(request):
    # The middleware already verified that this is a valid user
    pb_number = request.session.get('pb_number')
    user_data = users_collection.find_one({"pb_number": pb_number})
    return render(request, 'main/profile.html', {'user_data': user_data})


def edit_user(request, user_id):
    # Check if authenticated user is admin
    pb_number = request.session.get('pb_number')
    user_data = users_collection.find_one({"pb_number": pb_number})
    
    if not user_data or user_data.get('role', '').lower() != 'admin':
        messages.error(request, 'Access denied. Admin access only.')
        return redirect('signin')
    
    # Implementation for editing users
    messages.info(request, 'Edit user functionality will be implemented soon.')
    return redirect('admin_dashboard')

def delete_user(request, user_id):
    # Check if authenticated user is admin
    pb_number = request.session.get('pb_number')
    user_data = users_collection.find_one({"pb_number": pb_number})
    
    if not user_data or user_data.get('role', '').lower() != 'admin':
        messages.error(request, 'Access denied. Admin access only.')
        return redirect('signin')
    
    # Implementation for deleting users
    messages.info(request, 'Delete user functionality will be implemented soon.')
    return redirect('admin_dashboard')