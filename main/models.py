from django.contrib.auth.models import AbstractUser
from django.db import models
from db_connection import db

users_collection = db['users']

class User(AbstractUser):
    ROLES = [
        ('admin', 'Admin'),
        ('user', 'User'),
    ]
    pb_number = models.CharField(max_length=20, unique=True)
    password = models.CharField(max_length=128)
    name = models.CharField(max_length=100)
    gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')])
    role = models.CharField(max_length=10, choices=ROLES)
    division = models.CharField(max_length=100, blank=True, null=True)
    department = models.CharField(max_length=100, blank=True, null=True)
    designation = models.CharField(max_length=100, blank=True, null=True)

    def save(self, *args, **kwargs):
        from django.contrib.auth.hashers import make_password
        self.password = make_password(self.password)
        super().save(*args, **kwargs)


class Training(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
    category = models.CharField(max_length=100, choices=[
        ('aerodynamics', 'Aerodynamics'),
        ('aviation', 'Aviation'),
        ('personality-development', 'Personality Development'),
        ('soft-skills', 'Soft Skills'),
        ('corporate', 'Corporate'),
        ('industry', 'Industry Knowledge'),
    ])
    status = models.CharField(max_length=50, choices=[
        ('active', 'Active'),
        ('upcoming', 'Upcoming'),
        ('completed', 'Completed'),
    ])
    instructor_name = models.CharField(max_length=255)
    instructor_pb_number = models.CharField(max_length=50)
    start_date = models.DateField()
    end_date = models.DateField()
    duration = models.CharField(max_length=50, blank=True)
    image = models.ImageField(upload_to='training_images/')

    def save(self, *args, **kwargs):
        # Calculate duration
        if self.start_date and self.end_date:
            duration_days = (self.end_date - self.start_date).days
            self.duration = f"{duration_days} days"
        super(Training, self).save(*args, **kwargs)

    def __str__(self):
        return self.title
    
class LLMDocument(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')
    upload_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ('human', 'Human'),
        ('ai', 'AI'),
        ('ref', 'Reference'),
    ]
    
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)