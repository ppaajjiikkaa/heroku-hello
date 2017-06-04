from django.shortcuts import render
from django.http import HttpResponse, HttpRequest

def index(request):
    return HttpResponse(HttpRequest.GET.get("pokus",""))

# Create your views here.
