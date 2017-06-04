#!/usr/bin/env python3
from imutils import contours
import numpy as np
import imutils
import cv2
import random
import base64

#funkce
def linear_pts(n, min, max):
    size = (max - min) / (n-1)
    ret = []
    for i in range(n):
        ret.append(min+i*size)
    return ret
    

def centroid_matrix_from_contours(cnts, wdth, hght):
    cx = []
    cy = []
    for c in cnts:
        M = cv2.moments(c)
        cx.append(float(M["m10"] / M["m00"]))
        cy.append(float(M["m01"] / M["m00"]))

    kcx, _ = kmeans(cx, linear_pts(wdth, np.amin(cx), np.amax(cx)))
    kcy, _ = kmeans(cy, linear_pts(hght, np.amin(cy), np.amax(cy)))
  
    pts = np.zeros((int(hght*wdth), 2), int)
    i=0
    for x in range(wdth):
        for y in range(hght):
            pts[i][0] = int(kcx[x])
            pts[i][1] = int(kcy[y])
            i+=1
    return pts
        

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([a for a in mu]) == set([a for a in oldmu]))
 
def kmeans(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, len(K))
    mu = K
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu.copy(), clusters.copy())
    return(mu, clusters)


##vlastni detekce
def detect(imgbase):
    #imgfile = "./Tdp/20170531_144851.min.jpg"
    #img = cv2.imread(imgfile)
    imgdecode = base64.b64decode(imgbase.replace("data:image/jpeg;base64,","").strip())
    npimg = np.fromstring(imgdecode, dtype=np.int8)
    img = cv2.imdecode(npimg, 1)
    img = cv2.resize(img, (720,960))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur (gray ,(5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts =cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts= cnts[0] if imutils.is_cv2() else cnts[1]
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                docCnt = approx
                break


    paper = four_point_transform(img, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    thresh = cv2.adaptiveThreshold(warped,50,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,35,10)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    qcnts = []

    wdts = []
    hgts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        
        if w > 20 and h > 20 and ar > 0.8 and ar < 1.2:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx)==4:
                qcnts.append(c)
                wdts.append(w)
                hgts.append(h)

    pts = centroid_matrix_from_contours(qcnts, 10, 15)
    recs = []
    offw = int(np.average(wdts)/2-4)
    offh = int(np.average(hgts)/2-4)

    for p in pts:
        recs.append([p[0]-offw, p[1]-offh, p[0]+offw, p[1]+offh])
        cv2.rectangle(paper, (p[0]-offw, p[1]-offh), (p[0]+offw, p[1]+offh), (0,255,0))

    thresh2 = cv2.threshold(warped, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    i=0
    crossed=[]
    for r in recs:
        trec = thresh[r[1]:r[3], r[0]:r[2]]
        recavg = np.average(trec)
        
        if recavg < 44:
            cv2.rectangle(paper, (r[0], r[1]), (r[2], r[3]), (0,0,255))
            crossed.append(i)
        i+=1

    return crossed
    #cv2.imshow("img",paper)








