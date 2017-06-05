from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
#import cv2
#import base64
#import numpy as np
import aio
import json

def join_list(list_to_join):
    return (', '.join(str(x) for x in list_to_join))

def index(request):
    #ret = "<html><body>"
    imgbase = request.body
    #ret += "<img src='"+imgbase+"'>"
    #ret += "<p>"+join_list(aio.detect(imgbase))+"</p>"
    #ret += "</body></html>"
    #ret = request.POST.get("pokus","")
    
    #return HttpResponse(ret)
    #ret = JsonResponse(json.dumps(aio.detect(imgbase)), safe=False)
    #ret = HttpResponse(json.dumps(aio.detect(imgbase)))
    ret = HttpResponse(aio.detect(imgbase))
    return ret

# Create your views here.
