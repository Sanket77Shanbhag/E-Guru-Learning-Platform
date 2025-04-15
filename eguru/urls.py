"""
URL configuration for eguru project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from main import views
from django.conf import settings
from django.conf.urls.static import static
from main.views import create_user, manage_trainings, manage_llm, ask_eguru

urlpatterns = [
    path('', views.home, name='home'),
    path('signin/', views.signin, name='signin'),
    path('signup/', views.signup, name='signup'),
    path('signin/', views.signout, name='signout'),
    path('admin/',views.admin_dashboard, name='admin_dashboard'),
    path('user/',views.user_dashboard, name='user_dashboard'),
    path('create_user/', create_user, name='create_user'),
    path('manage_trainings/', manage_trainings, name='manage_trainings'),
    path('manage_llm/', manage_llm, name='manage_llm'),
    path('ask/', ask_eguru, name='ask_eguru'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
