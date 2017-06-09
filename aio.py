#!/usr/bin/env python3
from imutils import contours
from math import hypot
import numpy as np
import imutils
import cv2
import random
import base64


#funkce
def approx_shape(cnt, perimult = 0.01):
    peri = cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, perimult * peri, True) #peri * 0.02

def unique_pts(cnt):
    a = np.ascontiguousarray(cnt)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def largest_contour(cnts):
    area = 0
    ret = None
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        aps = approx_shape(c)
        cfp = contour_to_four_pts(aps)
        up = unique_pts(cfp)
        if w * h > area and len(up) == 4:
            ret = c
            area = w * h
    return ret

##def widest_contour(cnts):
##    hwr = 0
##    ret = None
##    print(len(cnts))
##    for c in cnts:
##        x,y,w,h = cv2.boundingRect(c)
##        if len(c) >= 4  and w / h > 3.5 and w / h < 4.5 and w / h > hwr:
##            print (w, h)
##            ret = c
##            hwr = w / h
##    return ret
    

def closest_point(pt, pts):
    ret = None
    mindist = None
    for p in pts:
        dist = hypot(p[0][0] - pt[0], p[0][1] - pt[1])
        if mindist == None: mindist = dist
        if mindist >= dist:
            mindist = dist
            ret = ([p[0][0], p[0][1]])
    return ret

def contour_to_four_pts(cnt):
    ret = np.zeros((4,2), np.int32)
    x,y,w,h = cv2.boundingRect(cnt)

    newpt = (x, y)
    ret[0] = closest_point(newpt, cnt)
    
    newpt = (x + w, y)
    ret[1] = closest_point(newpt, cnt)

    newpt = (x, y + h)
    ret[2] = closest_point(newpt, cnt)
    
    newpt = (x + w, y + h)
    ret[3] = closest_point(newpt, cnt)

    return ret

def linear_pts(n, min, max):
    size = (max - min) / (n-1)
    ret = []
    for i in range(n):
        ret.append(min+i*size)
    return ret
    

def min_diff(nparr):
    ret = None
    for i1 in range(len(nparr)):
        for i2 in range(len(nparr)):
            if i1 == i2: continue
            if ret == None:
                ret = abs(nparr[i1]-nparr[i2])
            else:
                ret = min(ret,abs(nparr[i1]-nparr[i2]))
    return ret

def centroid_matrix_from_contours(cnts, wdth, hght, img_shape):
    median_offset = 2 # vůle pro porovnání přesahu kontůr jednotlivých (X) políček
    edge_offset = 2 # zvětšení vzdálenosti od okraje obrázku    
    bottom_box_ratio = 1.1 # kolik procent dělá políčko splněné úkoly
    cx = []
    cy = []
    cw = []
    ch = []
    for c in cnts:
        M = cv2.moments(c)
        _, _, w, h = cv2.boundingRect(c)
        cw.append(w)
        ch.append(h)
        cx.append(float(M["m10"] / M["m00"]))
        cy.append(float(M["m01"] / M["m00"]))

   
    kcx, _ = kmeans(cx, linear_pts(wdth, np.amin(cx), np.amax(cx)))
    kcy, _ = kmeans(cy, linear_pts(hght, np.amin(cy), np.amax(cy)))

    mcw = int(np.median(cw))
    mch = int(np.median(ch))

    #redistribuce kmeans
    if len(kcx) != wdth or len(kcy) != hght or min_diff(kcx) < (mcw - median_offset) or min_diff(kcy) < (mch - median_offset):
        imgw = img_shape[1]
        imgh = img_shape[0] / bottom_box_ratio
        img_rem_w = imgw - (wdth * mcw)
        img_rem_h = imgh - (hght * mch)
        xdist = (img_rem_w - (2 * edge_offset)) / (wdth + 1)
        ydist = (img_rem_h - (2 * edge_offset)) / (hght + 1)
        x_center_offset = xdist + edge_offset + (mcw / 2)
        y_center_offset = ydist + edge_offset + (mch / 2)

        kcx = linear_pts(wdth, x_center_offset, imgw - x_center_offset)
        kcy = linear_pts(hght, y_center_offset, imgh - y_center_offset)
        
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
    ret = None
    imgOrig = base64toImg(imgbase)
    for i in range(4):
        ret = None
        img = imutils.rotate_bound(imgOrig, i*90) #dát i
        img = cv2.resize(img, (720,960))
        #cv2.imshow(str(i),img)
        #ret = detectSingle(img)
        try:
            ret = detectSingle(img)
            #print ("T",i,ret,np.average(ret))
        except:
            ret = False
            #print("E",i,ret,np.average(ret))
        if ret != False: break
    return ret

def base64toImg(imgbase):
    imgdecode = base64.b64decode(imgbase.replace("data:image/jpeg;base64,","").strip())
    npimg = np.fromstring(imgdecode, dtype=np.int8)
    return cv2.imdecode(npimg, 1)
    

def detectSingle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur (gray ,(5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred,50,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,35,10)
    edged = cv2.Canny(thresh, 75, 200)
    #cv2.imshow("E", edged)
    
    cnts =cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts= cnts[0] if imutils.is_cv2() else cnts[1]
    docCnt = None

    #cv2.drawContours(img, cnts, -1, (0,255,0), 1)
    #cv2.imshow("cnts_img",img)

    lcnt = largest_contour(cnts)
    approx = approx_shape(lcnt)
    docCnt = contour_to_four_pts(approx)

    #approximg = img.copy()
    #cv2.drawContours(approximg, docCnt, -1, (0,255,0), 1)
    #cv2.polylines(approximg, [docCnt], 1, (0,255,255))
    #cv2.imshow("cnts_img",approximg)

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

   
    #cv2.drawContours(paper, qcnts, -1, (0,255,0), 1)
    #cv2.imshow("cnts_warped",paper)
    
    pts = centroid_matrix_from_contours(qcnts, 10, 15, paper.shape)
    recs = []
    offw = int(np.average(wdts)/2-4)
    offh = int(np.average(hgts)/2-4)

    for p in pts:
        recs.append([p[0]-offw, p[1]-offh, p[0]+offw, p[1]+offh])
        #cv2.rectangle(paper, (p[0]-offw, p[1]-offh), (p[0]+offw, p[1]+offh), (0,0,255))
        
    #cv2.imshow("rects",paper)
    thresh = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,35,10)#35,10)
    #thresh2 = cv2.threshold(thresh.copy(), 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #cv2.imshow("img",thresh)
    i=0
    crossed=[]
    for r in recs:
        trec = thresh[r[1]:r[3], r[0]:r[2]]
        recavg = np.average(trec)

        #print(i, recavg)
        if recavg < 110 and recavg > 35:
            cv2.rectangle(paper, (r[0], r[1]), (r[2], r[3]), (0,0,255))
            crossed.append(i)
        i+=1

    return crossed
    







