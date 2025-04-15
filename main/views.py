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
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.contrib.auth.decorators import login_required
from .models import ChatSession, ChatMessage
import requests
import json
import numpy as np
import fitz

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['eguru']
users_collection = db['users']
trainings_collection = db['trainings']
llm_collection = db['document']
collection = db['eguru-llm-index']

embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
    all_trainings = list(trainings_collection.find({}))
    return render(request, 'home.html' , {'all_trainings': all_trainings})

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
                    image_directory = os.path.join('main','static' ,'training_images')
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
                        "image_url": f"training_images\{image.name}"
                    }
                    trainings_collection.insert_one(training_data)
                    messages.success(request, 'Training added successfully.')
                    return redirect('admin_dashboard')

    return render(request, 'admin_dashboard.html')

def manage_llm(request):
    if request.method == "POST" and request.FILES.get("document"):
        document = request.FILES["document"]
        document_name = document.name.split('.')[0]

        if llm_collection.find_one({"name": document_name}):
            messages.error(request, 'Document with this name already exists.')
        elif not document:
            messages.error(request, 'Please upload a document.')
        else:
            # Validate file size (Max 100MB)
            if document.size > 100 * 1024 * 1024:
                messages.error(request, 'File size exceeds 100MB.')
            else:
                # Validate file type
                valid_extensions = ['pdf', 'ppt', 'pptx', 'doc', 'docx']
                if document.name.split('.')[-1].lower() not in valid_extensions:
                    messages.error(request, 'Invalid file format.')
                else:
                    # Ensure directory exists
                    document_directory = os.path.join('main','static','llm_documents')
                    os.makedirs(document_directory, exist_ok=True)

                    # Save document to media folder
                    document_path = os.path.join(document_directory, document.name)
                    with open(document_path, 'wb+') as destination:
                        for chunk in document.chunks():
                            destination.write(chunk)

                    # Save to MongoDB
                    document_data = {
                        "name": document_name,
                        "type": document.name.split('.')[-1].upper(),
                        "upload_date": datetime.now(),
                        "file_url": f"llm_documents\{document.name}"
                    }
                    llm_collection.insert_one(document_data)
                    messages.success(request, 'Document uploaded successfully.')
                    return redirect('admin_dashboard')

    return render(request, 'admin_dashboard.html')

# Admin Dashboard with session validation
def admin_dashboard(request):
    # The middleware already verified that this is a valid admin user
    pb_number = request.session.get('pb_number')
    user_data = users_collection.find_one({"pb_number": pb_number})
    all_users = list(users_collection.find({}))
    all_trainings = list(trainings_collection.find({}))
    uploaded_documents = list(llm_collection.find({}))
    for doc in uploaded_documents:
        doc["id"] = str(doc["_id"])
    return render(request, 'admin_dashboard.html', {'all_users': all_users, 'admin': user_data, 'all_trainings': all_trainings, "uploaded_documents": uploaded_documents})

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


# Semantic Search
def find_relevant_chunks(query, top_k=5):
    query_emb = embedder.encode(query)
    all_docs = list(collection.find({}))
    for doc in all_docs:
        doc["score"] = np.dot(doc["embedding"], query_emb)
    sorted_docs = sorted(all_docs, key=lambda x: x["score"], reverse=True)
    return sorted_docs[:top_k]

# Call Ollama with context
def ask_ollama_with_context(context_chunks, user_query):
    context_text = "\n\n".join([
        f"[{doc['filename']} - Page {doc['page_number']}]\n{doc['text']}"
        for doc in context_chunks
    ])

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided document context to answer the question."},
        {"role": "user", "content": f"Context:\n{context_text}"},
        {"role": "user", "content": user_query}
    ]

    response = requests.post(
        "http://127.0.0.1:8000/ask/",
        json={"model": "llama3", "messages": messages, "stream": False}
    )

    return response.json()["message"]["content"]

# Django view to handle chat request
@csrf_exempt
def ask_eguru(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            query = data.get("query", "").strip()

            if not query:
                return JsonResponse({"response": "Please provide a valid question."}, status=400)

            chunks = find_relevant_chunks(query)
            answer = ask_ollama_with_context(chunks, query)

            return JsonResponse({"response": answer})

        except Exception as e:
            return JsonResponse({"response": f"❌ Error: {str(e)}"}, status=500)

    return JsonResponse({"response": "Invalid request method"}, status=405)