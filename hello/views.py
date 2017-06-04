from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
import cv2
import base64
import numpy as np
import aio

def join_list(list_to_join):
    return (', '.join(str(x) for x in list_to_join))

def index(request):
    ret = "<html><body>"
    imgbase = request.POST.get("pokus","")
    ret += "<img src='"+imgbase+"'>"
    ret += "<p>"+join_list(aio.detect(imgbase))+"</p>"
    ret += "</body></html>"
    #ret = request.POST.get("pokus","")
    
    return HttpResponse(ret)

# Create your views here.
