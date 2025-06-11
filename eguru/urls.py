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
from main.views import manage_trainings, manage_llm, ask_eguru

urlpatterns = [
    path('', views.home, name='home'),
    path('signin/', views.signin, name='signin'),
    path('signup/', views.signup, name='signup'),
    path('signin/', views.signout, name='signout'),
    path('change_password/', views.change_password, name='change_password'),
    path('admin/',views.admin_dashboard, name='admin_dashboard'),
    path('user/',views.user_dashboard, name='user_dashboard'),
    path('create_user/', views.create_user, name='create_user'),
    path('admin/import-users-excel/', views.import_users_excel, name='import_users_excel'),
    path('admin/delete-user/', views.delete_user, name='delete_user'),
    path('training/<str:title>/', views.training_details, name='training_details'),
    path('add_materials/', views.add_materials, name='add_materials'),
    path('delete_material/', views.delete_material, name='delete_material'),
    path('complete_training/', views.complete_training, name='complete_training'),
    path('manage_trainings/', manage_trainings, name='manage_trainings'),
    path('enroll/<str:training_id>/', views.enroll_training, name='enroll_training'),
    path('delete_training/', views.delete_training, name='delete_training'),
    path('manage_llm/', manage_llm, name='manage_llm'),
    path('ask/', ask_eguru, name='ask_eguru'),
    path('generate-quiz/', views.generate_quiz, name='generate_quiz'),
    path('save_quiz/', views.save_quiz, name='save_quiz'),
    path('accept/', views.accept, name='accept'),
    path('reject/', views.reject, name='reject'),
    path('submit-review/', views.submit_review, name='submit_review'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
