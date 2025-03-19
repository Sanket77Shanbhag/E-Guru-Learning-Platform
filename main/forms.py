from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User

class UserRegistrationForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['pb_number', 'password', 'name', 'gender', 'role', 'division', 'department', 'designation']

class LoginForm(forms.Form):
    pb_number = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)