from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import aio
import json
import cloud

def join_list(list_to_join):
    return (', '.join(str(x) for x in list_to_join))

def index(request):
    imgbase = request.body.decode("utf-8")
    cloud.cloudupload(imgbase)
    ret = JsonResponse(json.dumps(aio.detect(imgbase)), safe=False)
       
    return ret

# Create your views here.
