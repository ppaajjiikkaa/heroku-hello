from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
import cv2
import base64
import numpy as np

def index(request):
    ret = "<html><body>"
    ret += "<img src='request.POST.get("pokus","")'>
    ret += </body></html>
    #ret = request.POST.get("pokus","")
    
    return HttpResponse(ret)

# Create your views here.
