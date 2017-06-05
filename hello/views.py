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
    imgbase = request.body.decode("utf-8")
    ret = JsonResponse(json.dumps(aio.detect(imgbase)), safe=False)
    #ret = HttpResponse(json.dumps(aio.detect(imgbase)))
    #ret = HttpResponse(aio.detect(imgbase))
    return ret

# Create your views here.
