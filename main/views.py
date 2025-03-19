from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login

from .forms import LoginForm
from .models import User

def signin(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            pb_number = form.cleaned_data['pb_number']
            password = form.cleaned_data['password']
            user = authenticate(request, username=pb_number, password=password)
            if user is not None:
                login(request, user)
                if user.role == 'admin':
                    return redirect('/admin/dashboard/')
                else:
                    return redirect('/user/dashboard/')
            else:
                messages.error(request, 'Invalid PB Number or Password.')
    else:
        form = LoginForm()
    return render(request, 'main/signin.html', {'form': form})

def signout(request):
    logout(request)
    return redirect('/')

def admin_dashboard(request):
    return render(request, 'main/admin_dashboard.html')

def user_dashboard(request):
    return render(request, 'main/user_dashboard.html')

def profile(request):
    return render(request, 'main/profile.html', {'user': request.user})