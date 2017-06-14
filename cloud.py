import cloudinary
import cloudinary.uploader
import cloudinary.api
import base64

def cloudupload(imgbase):
    cloudinary.config(cloud_name = "pajika", api_key = "673854751339913", api_secret = "Mjm-0PYbcowb2s846BmUbC1K5QM")
    cloudinary.uploader.upload(imgbase)
