from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User, Training

class UserRegistrationForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['pb_number', 'password', 'name', 'gender', 'role', 'division', 'department', 'designation']

class LoginForm(forms.Form):
    pb_number = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

class TrainingForm(forms.ModelForm):
    class Meta:
        model = Training
        fields = ['title', 'description', 'category', 'status', 'instructor_name', 
                  'instructor_pb_number', 'start_date', 'end_date', 'image']
        widgets = {
            'start_date': forms.DateInput(attrs={'type': 'date'}),
            'end_date': forms.DateInput(attrs={'type': 'date'}),
        }