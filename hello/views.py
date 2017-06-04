from django.shortcuts import render
from django.http import HttpResponse, HttpRequest

def index(request):
    return HttpResponse(request.POST.get("pokus",""))

# Create your views here.
