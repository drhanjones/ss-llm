from django.shortcuts import render
from django import forms
from django.views.generic.edit import CreateView
# Create your views here.
from django.http import HttpResponse
from model_eval.models import TrainedModel

def index(request):
    return HttpResponse("Hello, world. You're at the model_eval index.")

class ModelForm(CreateView):
    model = TrainedModel
    fields = ['unique_id', 'name', 'description', 'n_layers', 'n_heads', 'n_embd']