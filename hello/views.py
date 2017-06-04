from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
import cv2

def index(request):
    ret = cv2.__version__
    #ret = request.POST.get("pokus","")
    return HttpResponse(ret)

# Create your views here.
