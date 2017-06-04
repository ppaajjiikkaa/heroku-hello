from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
import cv2
import base64
import numpy as np

def index(request):
    ret = "<html><body>"
    imgurl = request.POST.get("pokus","")
    ret += "<img src='"+imgurl+"'>"
    ret += "</body></html>"
    #ret = request.POST.get("pokus","")
    
    return HttpResponse(ret)

# Create your views here.
