from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password, check_password
from .forms import LoginForm, UserRegistrationForm, TrainingForm
from .models import User, users_collection, Training
from pymongo import MongoClient
from datetime import datetime
import os

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['eguru']
users_collection = db['users']
trainings_collection = db['trainings']

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

    return render(request, 'signin.html')

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

    return render(request, 'signup.html')


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
    return render(request, 'home.html')

def manage_trainings(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        description = request.POST.get('description')
        category = request.POST.get('category')
        status = request.POST.get('status')
        instructor_pb_number = request.POST.get('instructor_pb_number')
        instructor_name = request.POST.get('instructor_name')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')
        image = request.FILES.get('image')

        if trainings_collection.find_one({"title": title, "instructor_pb_number": instructor_pb_number}):
            messages.error(request, 'Training with this title and instructor already exists.')
        elif not image:
            messages.error(request, 'Please upload an image.')
        else:
            # Validate image size (Max 2MB)
            if image.size > 2 * 1024 * 1024:
                messages.error(request, 'Image size exceeds 2MB.')
            else:
                # Validate image format
                valid_extensions = ['png', 'jpg', 'jpeg', 'webp']
                if image.name.split('.')[-1].lower() not in valid_extensions:
                    messages.error(request, 'Invalid image format.')
                else:
                    # Ensure the directory exists
                    image_directory = os.path.join('training_images')
                    os.makedirs(image_directory, exist_ok=True)

                    # Save image to static folder
                    image_path = os.path.join(image_directory, image.name)
                    with open(image_path, 'wb+') as destination:
                        for chunk in image.chunks():
                            destination.write(chunk)

                    # Calculate duration
                    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                    duration_days = (end_date_obj - start_date_obj).days

                    # Save to MongoDB
                    training_data = {
                        "title": title,
                        "description": description,
                        "category": category,
                        "status": status,
                        "instructor_pb_number": instructor_pb_number,
                        "instructor_name": instructor_name,
                        "start_date": start_date_obj,
                        "end_date": end_date_obj,
                        "duration": f"{duration_days} days",
                        "image_url": image_path
                    }
                    trainings_collection.insert_one(training_data)
                    messages.success(request, 'Training added successfully.')
                    return redirect('admin_dashboard')

    return render(request, 'admin_dashboard.html')

# Admin Dashboard with session validation
def admin_dashboard(request):
    # The middleware already verified that this is a valid admin user
    pb_number = request.session.get('pb_number')
    user_data = users_collection.find_one({"pb_number": pb_number})
    all_users = list(users_collection.find({}))
    all_trainings = list(trainings_collection.find({}))
    return render(request, 'admin_dashboard.html', {'all_users': all_users, 'admin': user_data, 'all_trainings': all_trainings})

def user_dashboard(request):
    pb_number = request.session.get('pb_number')
    user_data = users_collection.find_one({"pb_number": pb_number})
    all_trainings = list(trainings_collection.find({}))
    return render(request, 'user_dashboard.html', {'user': user_data, 'all_trainings': all_trainings})

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
    return render(request, 'admin_dashboard.html')


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