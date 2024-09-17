from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('model_form/', views.ModelForm.as_view(), name='model_form'),
]
