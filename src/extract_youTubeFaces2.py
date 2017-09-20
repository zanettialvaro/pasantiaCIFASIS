# -*- coding: utf-8 -*-
#!/usr/bin/python

from __future__ import print_function
import dlib
#~ import openface
import cv2
import sys
import os
import multiprocessing
import time
import cPickle
import locale
import re
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
#~ from skimage.measure import compare_ssim as ssim  #http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics.pairwise import euclidean_distances
#~ from tqdm import tqdm
import random
import copy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.svm import SVC
import datetime

splineFig = plt.figure(1,figsize=(20,10))
DETECTION_THR = 0

def flattenList(l):
    return [item for sublist in l for item in sublist]

def euclidean(x128, y128):
    s = 0
    for i in range(0,128):
        s = s + pow((x128[i]-y128[i]),2.0);
    return np.sqrt(s);

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def listFiles(dire, rootCall = True):    #ext can be jpg|jpeg|bmp|png
    '''
    return a list with all the image files (png,jpg) inside the directory 'dire'
    '''
    filesJPGRoot = [os.path.join(dire, f) for f in os.listdir(dire)
                if os.path.isfile(os.path.join(dire, f)) and (".jpg" in f)]
    filesPNGRoot = [os.path.join(dire, f) for f in os.listdir(dire)
                if os.path.isfile(os.path.join(dire, f)) and (".png" in f)]
    dires = [os.path.join(dire, f) for f in os.listdir(dire)
             if os.path.isdir(os.path.join(dire, f))]
    filesInDirs = []
    while dires:
        d = dires.pop()
        filesInD = listFiles(d,False)
        filesInDirs += filesInD
    files = filesJPGRoot + filesPNGRoot + filesInDirs
    files.sort(key=natural_key)
    if rootCall:
        print("{} archivos encontrados con ext=\".jpg\" y \".png\" en: {}".format(len(files),dire))
    return files

def pathCheck(path):
    if path and not os.path.exists(path):
        os.makedirs(path)
    return path

def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)

def overlap(r1, r2):
    '''Overlapping rectangles overlap both horizontally & vertically
    '''
    return (range_overlap(r1.left(), r1.right(), r2.left(), r2.right())
           and
           range_overlap(r1.top(), r1.bottom(), r2.top(), r2.bottom()))

def put_text_in_rect(img,text,rect,color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (rect.left(),rect.bottom()+20)#Bottom-left corner of the text string in the image.
    fontScale = 0.7 #Font scale factor that is multiplied by the font-specific base size.
    thickness = 2
    cv2.putText(img,text,org,font,fontScale,color,thickness)

def draw_rectangle(img, myRect, color=(255,255,255), thickness=2):
    cv2.rectangle(img,(myRect.get_rect().left(),myRect.get_rect().top()),(myRect.get_rect().right(),myRect.get_rect().bottom()),color,thickness)

def draw_landmark(img,points,color,thickness):
    for i in range(0,len(points)-1):
        cv2.line(img, points[i], points[i+1], color, thickness=thickness)

def draw_landmarks(img,lands,score,color,thickness):
    points = [(p.x,p.y) for p in lands.parts()]
    chin = points[0:17]
    eyebrow1 = points[17:22]
    eyebrow2 = points[22:27]
    nose = points[27:36]+[points[30]]
    eye1 = points[36:42]+[points[36]]
    eye2 = points[42:48]+[points[42]]
    mouth = points[48:68]+[points[60]]
    for part in [chin,eyebrow1,eyebrow2,nose,eye1,eye2,mouth]:
        draw_landmark(img,part,color,thickness)
    r = lands.rect
    cv2.rectangle(img,(r.left(),r.top()),(r.right(),r.bottom()),color,thickness=thickness)
    if score != None:
        put_text_in_rect(img, str(round(score,4)), r, color)

def show_img(filepath):
    img=cv2.imread(filepath)
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def show_face(face):
    img = cv2.imread(face.get_inImage().get_filepath())
    myR = face.get_myRect()
    rect = myR.get_rect()
    l,t,r,b = rect.left(), rect.top(), rect.right(), rect.bottom()
    face = img[t:b+1, l:r+1]
    cv2.imshow('',face)
    cv2.waitKey()
    cv2.destroyAllWindows()

def save_face(face,savePath):
    imgPath = face.get_inImage().get_filepath()
    #~ print("guardando face: {}".format(imgPath))
    facePath = os.path.join(pathCheck(savePath),os.path.basename(imgPath))
    img = cv2.imread(imgPath)
    myR = face.get_myRect()
    rect = myR.get_rect()
    rect = resize_negative_bounds(rect, img_height=256, img_width=256)
    l,t,r,b = rect.left(), rect.top(), rect.right(), rect.bottom()
    if r-l < 128 or b-t < 128:
        print("save_face(): no se guardará {} por ser muy pequeña".format(os.path.basename(facePath)))
    else:
        face = img[t:b+1, l:r+1]
        cv2.imwrite(facePath,face)

def resize_negative_bounds(rect, img_height=800, img_width=1920):
    l,t,r,b = rect.left(),rect.top(),rect.right(),rect.bottom()
    l = max(0,l)
    t = max(0,t)
    r = min(r,img_width)
    b = min(b,img_height)
    return dlib.rectangle(l,t,r,b)

class myRect(object):
    def __init__(self,dlibRect,score,interpolated):
        self.r = dlibRect
        self.interpolated = interpolated
        self.score = score

    def get_rect(self):
        return self.r
    def get_score(self):
        return self.score
    def get_center(self):
        return self.r.center()
    def get_height(self):
        return self.r.height()
    def get_width(self):
        return self.r.width()
    def get_max_square(self):
        return max(self.r.height(),self.r.width())
    def get_area(self):
        return self.r.area()

    def is_interpolated(self):
        return self.interpolated

    def __eq__(self,other):
        if other == None:
            return False
        return self.__dict__ == other.__dict__
    def __ne__(self,other):
        return not(self.__eq__(other))

    def __str__(self):
        return str(self.__dict__)

class myFace(object):
    def __init__(self,myR,lands,emb,inImage):
        self.faceID = None
        self.myRect = myR
        self.inImage = inImage #'parent' image
        self.lands = lands
        self.emb = emb
        self.smoothRect = None

        self.fillLeft = 0
        self.fillTop = 0
        self.fillRight = 0
        self.fillBottom = 0
        self.fillTrk = None

        self.tooSmall = False
        self.relevant = False

    def get_id(self):
        return self.faceID
    def set_id(self,faceID):
        self.faceID = faceID
    def get_myRect(self):
        return self.myRect
    def get_lands(self):
        return self.lands
    def get_emb(self):
        return self.emb
    def get_inImage(self):
        return self.inImage
    def get_tooSmall(self):
        return self.tooSmall
    def set_tooSmall(self):
        self.tooSmall = True
    def get_relevant(self):
        return self.relevant
    def set_relevant(self):
        self.relevant = True

    def get_fills(self):
        return (self.fillLeft,self.fillTop,self.fillRight,self.fillBottom)
    def set_fillLeft(self,n):
        self.fillLeft = n
    def set_fillTop(self,n):
        self.fillTop = n
    def set_fillRight(self,n):
        self.fillRight = n
    def set_fillBottom(self,n):
        self.fillBottom = n
    def set_fillTrack(self,fillTrk):
        self.fillTrk = fillTrk
    def get_fillTrack(self):
        return self.fillTrk

    def is_interpolated(self):
        return self.myRect.is_interpolated()

    def set_smoothRect(self,sRect):
        self.smoothRect = sRect
    def get_smoothRect(self):
        return self.smoothRect

class myImage(object):
    def __init__(self,frameIx,filepath):
        self.frameIx = frameIx
        self.filepath = filepath
        self.nDetected = 0
        self.nInterpolated = 0
        self.faces = []
        self.processTime = None

    def get_frameIx(self):
        return self.frameIx
    def get_filepath(self):
        return self.filepath
    def get_nDetected(self):
        return self.nDetected
    def get_nInterpolated(self):
        return self.nInterpolated
    def get_faces(self):
        return self.faces

    def process(self,upscale=0):
        #~ print("Procesando imagen: '{}'".format(self.filepath))
        st = time.time()
        img = cv2.imread(self.filepath)
        #~ upscale = 0
        dets,scores,idx = detector.run(img, upscale, DETECTION_THR)
        if len(dets) == 0: #reintentar aplicando CLAHE
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_clahe = clahe.apply(gray)
            dets,scores,idx = detector.run(img_clahe, upscale, DETECTION_THR)
        if len(dets) > 0:
            for r,s,fID in zip(dets,scores,range(0,len(dets))):
                myR = myRect(r,s,False)
                lands = face_pose_predictor(img, myR.get_rect())
                emb = embedder.compute_face_descriptor(img, lands)
                face = myFace(myR,lands,emb,self)
                self.faces.append(face)
                self.nDetected += 1
        self.processTime = time.time() - st

class faceGroup(object):
    def __init__(self,fgID):
        self.fgID = fgID
        self.faceID = None
        self.status = "No procesado (vacío)"
        self.faces = []
        self.imgsToInterpolate = []
        #~ self.minFIx = None
        #~ self.maxFIx = None
        self.lDet = None
        self.hDet = None

        self.xSplineP27 = None
        self.ySplineP27 = None
        self.xSplineRect = None
        self.ySplineRect = None
        self.hSpline = None
        self.wSpline = None
        self.dSpline = None

        self.scene_low = None
        self.scene_high = None
        self.last_face = None

    def get_id(self):
        return self.fgID
    def get_status(self):
        return self.status

    def get_faceID(self):
        return self.faceID
    def set_faceID(self,faceID):
        self.faceID = faceID

    def get_scene_low_high(self):
        return (self.scene_low,self.scene_high)
    def set_scene_low_high(self,l,h):
        self.scene_low = l
        self.scene_high = h

    def get_faces(self):
        return self.faces

    def is_empty(self):
        return (self.last_face == None)

    def overlap(self,face):
        ol = overlap(self.last_face.get_myRect().get_rect(),face.get_myRect().get_rect())
        lastImgFrame = self.last_face.get_inImage().get_frameIx()
        faceFrame = face.get_inImage().get_frameIx()
        frameDist = np.abs(lastImgFrame - faceFrame) < 20
        return (ol and frameDist)

    def image_here(self,img):
        for face in self.faces:
            if face.get_inImage().get_frameIx() == img.get_frameIx():
                return True
        return False

    def add_imgToInterpolate(self,img):
        self.imgsToInterpolate.append(img)
    def add_face(self,face):
        self.faces.append(face)
        self.last_face = face
    def calculate_detLimits(self):
        #~ self.minFIx = min([img.get_frameIx() for img in self.imgsToInterpolate + [f.get_inImage() for f in self.faces]])
        #~ self.maxFIx = max([img.get_frameIx() for img in self.imgsToInterpolate + [f.get_inImage() for f in self.faces]])
        if not(self.is_empty()):
            self.lDet = min([img.get_frameIx() for img in [f.get_inImage() for f in self.faces]])
            self.hDet = max([img.get_frameIx() for img in [f.get_inImage() for f in self.faces]])
            #~ print("fgID:{} - lDet,hDet = {},{}".format(self.fgID,self.lDet,self.hDet))
    def get_prev_next_det(self,fIx):
        #~ print("fIx: {} - lDet: {} - hDet: {}".format(fIx, self.lDet, self.hDet))
        fIxs = [f.get_inImage().get_frameIx() for f in self.faces]
        prevs = [f for f in fIxs if f < fIx]
        nexts = [f for f in fIxs if f > fIx]
        if prevs and nexts:
            return max(prevs),min(nexts)
        else:
            print("\nget_prev_next_det(): no hay suficientes datos.")

    def calculate_splines(self, plot = False):
        #~ print("\n==============faceGroup id: {}==============".format(self.fgID))
        centersRect = [f.get_myRect().get_center() for f in self.faces]
        centersP27 = [f.get_lands().part(27) for f in self.faces]
        xCensP27 = [c.x for c in centersP27]
        yCensP27 = [c.y for c in centersP27]
        xCensRect = [c.x for c in centersRect]
        yCensRect = [c.y for c in centersRect]
        dists = []
        for ix,f in enumerate(self.faces):
            d = f.get_lands().part(33).y - centersP27[ix].y
            dists.append(d)
        heights = [f.get_myRect().get_height() for f in self.faces]
        widths = [f.get_myRect().get_width() for f in self.faces]
        t = [f.get_inImage().get_frameIx() for f in self.faces]


        m = len(t)
        #~ sPos = m/2
        #~ sSize = m
        sPos = 5*m
        sSize = 1000*m
        #~ k = 3
        k = 2
        if len(t) < 2:
            #~ print("No se puede realizar la spline con menos de 2 datos. (fgID: {})".format(self.fgID))
            self.status = "Cant. de datos insuficientes"
            return
        elif len(t) == 2 :
            #~ print("Hay sólo 2 datos, se realizará una spline de grado 1. (fgID: {})".format(self.fgID))
            k = 1
        #~ elif len(t) == 3:
            #~ print("Hay sólo 3 datos, se realizará una spline de grado 2. (fgID: {})".format(self.fgID))
            #~ k = 2
        self.xSplineRect = interpolate.splrep(t,xCensRect,s=sPos,k=k)
        self.ySplineRect = interpolate.splrep(t,yCensRect,s=sPos,k=k)
        self.xSplineP27 = interpolate.splrep(t,xCensP27,s=sPos,k=k)
        self.ySplineP27 = interpolate.splrep(t,yCensP27,s=sPos,k=k)
        self.hSpline = interpolate.splrep(t,heights,s=sSize,k=k)
        self.wSpline = interpolate.splrep(t,widths,s=sSize,k=k)
        self.dSpline = interpolate.splrep(t,dists,s=sSize,k=k)

        self.status = "Splines OK"

        if plot:
            #~ splineRange   = range(max(self.lDet-3,self.minFIx),min(self.hDet+3,self.maxFIx)+1)
            splineRange   = range(self.lDet,self.hDet+1)
            splineXCensRect = interpolate.splev(splineRange, self.xSplineRect, der=0)
            splineYCensRect = interpolate.splev(splineRange, self.ySplineRect, der=0)
            splineXCensP27  = interpolate.splev(splineRange, self.xSplineP27, der=0)
            splineYCensP27  = interpolate.splev(splineRange, self.ySplineP27, der=0)
            splineHeights = interpolate.splev(splineRange, self.hSpline, der=0)
            splineWidths  = interpolate.splev(splineRange, self.wSpline, der=0)
            splineDists   = interpolate.splev(splineRange, self.dSpline, der=0)

            processed_path = os.path.join("./plots/faceGroups/",set_name)
            fgPath = '/'.join(self.faces[0].get_inImage().get_filepath().split('/')[-3:-1])
            splineFig = plt.figure(1)
            plt.suptitle('faceGroup id: {} - cant datos: {} - spline grad: {} - sPos = {} - sSizr = {}\npath = {}'.format(self.fgID, len(t),k,sPos,sSize,fgPath))

            plt.subplot(2,2,1)
            plt.plot(t,xCensRect,':bo',splineRange,splineXCensRect,'-c',t,yCensRect,':ro',splineRange,splineYCensRect,'-m',linewidth=1,markersize=2)
            plt.grid(True)
            plt.legend(['xCenRectReal', 'xCenSplineRect','yCenRectReal', 'yCenSplineRect'],loc='best',ncol=4,fontsize=7)

            plt.subplot(2,2,2)
            plt.plot(t,xCensP27,':bo',splineRange,splineXCensP27,'-c',t,yCensP27,':ro',splineRange,splineYCensP27,'-m',linewidth=1,markersize=2)
            plt.grid(True)
            plt.legend(['xCenP27Real', 'xCenSplineP27','yCenP27Real', 'yCenSplineP27'],loc='best',ncol=4,fontsize=7)

            plt.subplot(2,2,3)
            plt.plot(t,heights,':bo',splineRange,splineHeights,'-c',t,widths,':ro',splineRange,splineWidths,'-m',linewidth=1,markersize=2)
            plt.grid(True)
            plt.legend(['hReal', 'hSpline','wReal', 'wSpline'],loc='best',ncol=4,fontsize=7)

            plt.subplot(2,2,4)
            plt.plot(t,dists,':bo',splineRange,splineDists,'-c',linewidth=1,markersize=2)
            plt.grid(True)
            plt.legend(['dReal', 'dSpline'],loc='best',ncol=4,fontsize=7)

            splineFig.savefig(os.path.join(pathCheck(processed_path),'faceGropID{}.png'.format(self.fgID)), bbox_inches='tight',dpi=300)
            splineFig.clf()
        #~ print('===========================================')

    def interpolate(self):
        if self.status == "Cant. de datos insuficientes":
            return
        for imgti in self.imgsToInterpolate:
            fIx = imgti.get_frameIx()
            xCenRect = interpolate.splev([fIx], self.xSplineRect, der=0)[0]
            yCenRect = interpolate.splev([fIx], self.ySplineRect, der=0)[0]
            h = interpolate.splev([fIx], self.hSpline, der=0)[0]
            w = interpolate.splev([fIx], self.wSpline, der=0)[0]

            l = int(round(xCenRect-w/2))
            t = int(round(yCenRect-h/2))
            r = int(round(xCenRect+w/2))
            b = int(round(yCenRect+h/2))

            cv2img = cv2.imread(imgti.get_filepath())
            rect = resize_negative_bounds(dlib.rectangle(l,t,r,b),img_height=cv2img.shape[0],img_width=cv2img.shape[1])
            myR = myRect(rect,-1,True)
            lands = face_pose_predictor(cv2img, myR.get_rect())
            emb = embedder.compute_face_descriptor(cv2img, lands)
            face = myFace(myR,lands,emb,imgti)
            face.set_id(self.faceID)
            self.faces.append(face)

            drawImg = copy.deepcopy(cv2img)
            draw_landmarks(drawImg,face.lands,None,red,thickness=3)
            cv2.imwrite(os.path.join(pathCheck('./interpolatedImgs/fg{}'.format(self.fgID)),os.path.basename(imgti.get_filepath())),drawImg)

    def smooth_faces(self):
        if self.status == "Cant. de datos insuficientes":
            return
        for face in self.faces:
            fIx = face.get_inImage().get_frameIx()
            cv2img = cv2.imread(face.get_inImage().get_filepath())
            img_height = cv2img.shape[0]
            img_width = cv2img.shape[1]

            xCenP27 = interpolate.splev([fIx], self.xSplineP27, der=0)[0]
            yCenP27 = interpolate.splev([fIx], self.ySplineP27, der=0)[0]
            d = interpolate.splev([fIx], self.dSpline, der=0)[0]

            multUp = 2.3#1.5
            multDown = 2.3#2.1

            t = int(round(yCenP27-multUp*d))
            b = int(round(yCenP27+multDown*d))
            smoothHeight = b-t
            spaceSide = smoothHeight/2.0
            l = int(round(xCenP27-spaceSide))
            r = int(round(xCenP27+spaceSide))

            if t < 0:
                face.set_fillTop(-t)
                t = 0
            if b > img_height:
                face.set_fillBottom(b-img_height)
                b = img_height
            if l < 0:
                face.set_fillLeft(-l)
                l = 0
            if r > img_width:
                face.set_fillRight(r-img_width)
                r = img_width

            rect = dlib.rectangle(l,t,r,b)
            sRect = myRect(rect,-2,False)
            face.set_smoothRect(sRect)

    def identify_face(self,svm):
        if self.status == "Cant. de datos insuficientes":
            return
        for face in self.faces:
            #check if face is relevant
            isRelevant = svm.predict(np.array(face.get_emb()).reshape(1,-1))[0] == 0
            #~ print("face {} is relevant: {}".format(os.path.basename(face.get_inImage().get_filepath()),isRelevant))
            if isRelevant:
                face.set_relevant()

    def save_smoothed_to_disk(self, cropped_dest, split_relevant = False, track_fills=False, draw_lands = False, folder_structure = False):
        if self.status == "Cant. de datos insuficientes":
            return
        for face in self.faces:
            path = face.get_inImage().get_filepath()
            sRect = face.get_smoothRect()
            rect = sRect.get_rect()
            l,t,r,b = rect.left(), rect.top(), rect.right(), rect.bottom()
            if r-l < 80 or b-t < 80:
                #~ print("\n\t|save_smoothed_to_disk(): (fgID: {}) {} (faceID: {} - interpolated: {}) muy pequeña (widht = {} - height = {}), no se guardará.".format(self.fgID,os.path.basename(path),face.get_id(),face.is_interpolated(),r-l,b-t))
                face.set_tooSmall()
                continue

            cv2img = cv2.imread(path)
            newPath = path.replace("test_sets",cropped_dest,1) #el primer argumento debe ser la carpeta donde están contenidos los datasets, de manera de guardar los resultados en un path similar donde se reemplaza test_sets por 'cropped_dest', se generará una carpeta 'cropped_dest' con las respectivas subcarpetas.
            folder,basename = os.path.split(newPath)
            if folder_structure:
                facePath = os.path.join(pathCheck(os.path.join(folder,"fgID{}".format(self.fgID))),basename)
            else:
                if split_relevant:
                    if face.get_relevant():
                        classFolder = "relevantID/"
                    else:
                        classFolder = "notRelevantID/"
                else:
                    classFolder = ""
                facePath = os.path.join(pathCheck(os.path.join(os.path.join(os.path.join("./",cropped_dest),classFolder),"fgID{}".format(self.fgID))),basename)

            if draw_lands:
                cen = face.lands.part(27)
                p33 = face.lands.part(33)
                cv2.circle(cv2img,(cen.x,cen.y),2,(0,255,0),2)
                cv2.circle(cv2img,(p33.x,p33.y),2,(0,0,255),2)
                if face.is_interpolated():
                    draw_landmarks(cv2img,face.lands,None,blue,thickness=2)
                else:
                    draw_landmarks(cv2img,face.lands,None,white,thickness=2)

            fillL, fillT, fillR, fillB = face.get_fills()
            croppedImg = cv2img[t:b+1, l:r+1]
            origShape = croppedImg.shape
            if fillT > 0:
                #~ print("fg: {} - fillT={} in img: {} with shape: {}".format(self.fgID,fillT,face.get_inImage().get_filepath(),croppedImg.shape))
                fillWith = croppedImg[0,:,:].reshape(1,croppedImg.shape[1],croppedImg.shape[2])
                topStripe = np.tile(fillWith,(fillT,1,1))
                croppedImg = np.vstack([topStripe,croppedImg])
            if fillB > 0:
                #~ print("fg: {} - fillB={} in img: {} with shape: {}".format(self.fgID,fillB,face.get_inImage().get_filepath(),croppedImg.shape))
                fillWith = croppedImg[croppedImg.shape[0]-1,:,:].reshape(1,croppedImg.shape[1],croppedImg.shape[2])
                bottomStripe = np.tile(fillWith,(fillB,1,1))
                croppedImg = np.vstack([croppedImg,bottomStripe])
            if fillL > 0:
                #~ print("fg: {} - fillL={} in img: {} with shape: {}".format(self.fgID,fillL,face.get_inImage().get_filepath(),croppedImg.shape))
                fillWith = croppedImg[:,0,:].reshape(croppedImg.shape[0],1,croppedImg.shape[2])
                leftStripe = np.tile(fillWith,(1,fillL,1))
                croppedImg = np.hstack([leftStripe,croppedImg])
            if fillR > 0:
                #~ print("fg: {} - fillR={} in img: {} with shape: {}".format(self.fgID,fillR,face.get_inImage().get_filepath(),croppedImg.shape))
                fillWith = croppedImg[:,croppedImg.shape[1]-1,:].reshape(croppedImg.shape[0],1,croppedImg.shape[2])
                rightStripe = np.tile(fillWith,(1,fillR,1))
                croppedImg = np.hstack([croppedImg,rightStripe])
            if track_fills:
                track = fillTrack(self.fgID,face.get_fills(),facePath,origShape)
                #~ fillTracks.append(track)
                face.set_fillTrack(track)

            if np.abs(croppedImg.shape[0] - croppedImg.shape[1]) > 1:
                print("\nsave_smoothed_to_disk(): fgID: {} - croppedImg: {} | origShape: {} - newShape: {}".format(self.fgID,os.path.basename(facePath),origShape,croppedImg.shape))
            hMax = 256
            wMax = 256
            #~ ratio = min(wMax/float(wIm), hMax/float(hIm))
            #~ wNew,hNew = int(wIm*ratio),int(hIm*ratio)
            resizedImg = cv2.resize(croppedImg,(wMax,hMax),interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(facePath,resizedImg)

class fillTrack(object):
    def __init__(self,fg,fills,facePath,origShape):
        self.fg = fg
        self.fillLeft = fills[0]
        self.fillTop = fills[1]
        self.fillRight = fills[2]
        self.fillBottom = fills[3]
        self.facePath = facePath
        self.origShape = origShape
        self.resShape = (origShape[0]+self.fillTop+self.fillBottom,origShape[1]+self.fillLeft+self.fillRight,origShape[2])
        #~ print("fg: {} - croppedImg: {} | origShape: {} - resShape: {}".format(self.fg,self.facePath,self.origShape,self.resShape))

    def get_fg(self):
        return self.fg
    def get_fills(self):
        return self.fills
    def get_facePath(self):
        return self.facePath
    def get_origShape(self):
        return self.origShape
    def get_resShape(self):
        return self.resShape

def group_faces_by_folder():
    #get all folders
    folders = []
    for img in imgs:
        folder,basename = os.path.split(img.get_filepath())
        if not(folder in folders):
            folders.append(folder)

    #group images
    fgID = 0
    allFaceGroups = []
    for currFolder in folders:
        max_detected = 0
        currFaceID = 0
        fGs = [faceGroup(fgID)]
        fGs[0].set_faceID(currFaceID)
        fgID += 1
        imgsInCurrFold = [img for img in imgs if currFolder == os.path.split(img.get_filepath())[0]]
        for img in imgsInCurrFold:
            if img.get_nDetected() > max_detected:
                max_detected = img.get_nDetected()
            for face in img.get_faces():
                accepted = False
                for fg in fGs:
                    if fg.is_empty() or (fg.overlap(face) and not(fg.image_here(face.get_inImage()))):
                        accepted = True
                        face.set_id(fg.get_faceID())
                        fg.add_face(face)
                        break
                if not(accepted):
                    fg_new = faceGroup(fgID)
                    currFaceID += 1
                    fg_new.set_faceID(currFaceID)
                    face.set_id(currFaceID)
                    fgID += 1
                    fg_new.add_face(face)
                    fGs.append(fg_new)

        for fg in fGs:
            fg.calculate_detLimits()

        for img in imgsInCurrFold:
            if img.get_nDetected() < max_detected:
                frameIx = img.get_frameIx()
                for fg in fGs:
                    if not(fg.is_empty()) and not(fg.image_here(img)) and fg.lDet < frameIx < fg.hDet:
                        prevF,nextF = fg.get_prev_next_det(frameIx)
                        if np.abs(frameIx-prevF)<=3 or np.abs(frameIx-nextF)<=3:
                            fg.add_imgToInterpolate(img)

        allFaceGroups = allFaceGroups + fGs
    return allFaceGroups

def within_range(percent,v1,v2):
    #~ print("v1={}, v2={}".format(v1,v2))
    return ((v2*(1-percent)) < v1 < (v2*(1+percent))
            or
            (v1*(1-percent)) < v2 < (v1*(1+percent)))

def similarity_hist((myImg1,myImg2)):
    path1 = myImg1.get_filepath()
    path2 = myImg2.get_filepath()

    img1 = cv2.cvtColor(cv2.imread(path1),cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread(path2),cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([img1],channels=[0],mask=None,histSize=[256],ranges=[0,256]).reshape(256,)
    hist2 = cv2.calcHist([img2],channels=[0],mask=None,histSize=[256],ranges=[0,256]).reshape(256,)

    return euclidean_distances([hist1,hist2])[0][1]

def delete_multiple_detections(imgs):
    multDetsFolders = []
    newImgs = []
    for img in imgs:
        if img.get_nDetected() > 1:
            folder,basename = os.path.split(img.get_filepath())
            if not(folder in multDetsFolders):
                multDetsFolders.append(folder)
    for img in imgs:
        folder,basename = os.path.split(img.get_filepath())
        if not(folder in multDetsFolders):
            newImgs.append(img)
    return newImgs

def test_in_youTubeFaces(path, cantToProcess = 'all', loadPickle = False, pickleIt = True, upscale = 0):
    if loadPickle:
        with open('./pickled_objs/imgsYT_{}_upscale{}.obj'.format(cantToProcess,upscale), 'rb') as fp:
            imgsYT = cPickle.load(fp)
            timesYT = cPickle.load(fp)
            fp.close()
    else:
        #~ filesYT = listFiles("/home/alvaro/Desktop/tesina/test_sets/YouTubeFacesPartial/")
        filesYT = listFiles(path)
        sys.stdout.write("Creando contenedores de imagenes...")
        sys.stdout.flush()
        if cantToProcess == 'all':
            imgsYT = [image(fIx,filesYT[fIx]) for fIx in range(0,len(filesYT))]
        else:
            imgsYT = [image(fIx,filesYT[fIx]) for fIx in range(0,len(filesYT[:cantToProcess]))]
        print("OK")

        print("Procesando imgs...")
        bar = progressbar.ProgressBar()
        timesYT = []
        for img in bar(imgsYT):
            timesYT.append(img.process(upscale=upscale))

        if pickleIt:
            with open('./pickled_objs/imgsYT_{}_upscale{}.obj'.format(cantToProcess,upscale), 'wb') as fp:
                cPickle.dump(imgsYT, fp)
                cPickle.dump(timesYT, fp)
                fp.close()

    return imgsYT

#-----------------------------------

    #~ imgsYT = delete_multiple_detections(imgsYT)
    #~ fGs = group_faces_by_folder(imgsYT)

    #~ print("Procesando fgID:")
    #~ for fg in fGs:
        #~ sys.stdout.write("\t| fgID: {}...".format(fg.get_id()))
        #~ sys.stdout.flush()
        #~ fg.calculate_min_max_fIx()
        #~ fg.calculate_splines(plot=True)
        #~ fg.add_interpolated()
        #~ fg.smooth_faces()
        #~ fg.save_smoothed_to_disk()
        #~ print("{} |".format(fg.get_status()))

#=======================================================================

red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
white = (255,255,255)

def pool_process_img(img):
    img.process()
    return img

def pool_process1_fg(fg):
    if not(fg.is_empty()):
        fg.calculate_splines(plot=True)
        fg.interpolate()
        fg.smooth_faces()
        fg.save_smoothed_to_disk(cropped_dest,draw_lands=False,track_fills=True,folder_structure=True) #args: cropped_dest, split_relevant = False, draw_lands = False, folderStructure = False
    return fg

if __name__ == '__main__':
    startTime = time.time()
    print()
    if len(sys.argv)<3:
        print("Modo de uso: python extract_youTubeFaces.py 'imgs_src_path' 'cropped_dest'\nEl dataset debe estar contenido dentro de una carpeta 'test_sets' ya que luego se reemplazará esa parte del path por 'cropped_dest'")
        sys.exit()

    if sys.argv[1][-1] == '/':
        set_name = os.path.split(sys.argv[1][:-1])[1]
    else:
        set_name = os.path.split(sys.argv[1])[1]

    detector = dlib.get_frontal_face_detector()
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    rec_model_v1 = "dlib_face_recognition_resnet_model_v1.dat"
    embedder = dlib.face_recognition_model_v1(rec_model_v1)

    cantToProcess = 'all'
    loadImgs = False

    cropped_dest = sys.argv[2]
    files = listFiles(sys.argv[1])
    if cantToProcess != 'all':
        print("Se procesarán sólo las primeras {} imgs".format(cantToProcess))
        files = files[:cantToProcess]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    if loadImgs:
        sys.stdout.write("Cargando imgs desde .obj...")
        sys.stdout.flush()
        with open("pickled_objs/"+set_name+'_imgs_detThr{}.obj'.format(DETECTION_THR), 'rb') as fp:
            imgs = cPickle.load(fp)
            fp.close()
        print("OK")
    else:
        imgsProcStart = time.time()
        sys.stdout.write("Creando contenedores de imagenes...")
        sys.stdout.flush()
        imgs = [myImage(fIx,files[fIx]) for fIx in range(0,len(files))]
        print("OK")

        sys.stdout.write("Procesando imgs...")
        sys.stdout.flush()
        imgs = pool.map(pool_process_img, imgs)
        print("OK")
        imgsProcEnd = time.time()

        sys.stdout.write("pickling imgs.obj...")
        sys.stdout.flush()
        with open("pickled_objs/"+set_name+'_imgs_detThr{}.obj'.format(DETECTION_THR), 'wb') as fp:
            cPickle.dump(imgs, fp)
            fp.close()
        print("OK")

    fgsProc1Start = time.time()
    fGs = group_faces_by_folder()
    print()
    sys.stdout.write("faceGroups: Interpolando, suavizando y guardando SIN identificar...")
    sys.stdout.flush()
    fGs = pool.map(pool_process1_fg, fGs)
    print("OK")
    fgsProc1End = time.time()

    sys.stdout.write("pickling fGs.obj...")
    sys.stdout.flush()
    with open("pickled_objs/"+set_name+"_fGs.obj", 'wb') as fp:
        cPickle.dump(fGs, fp)
        fp.close()
    print("OK")

    sys.stdout.write("pickling fillTracks.obj...")
    sys.stdout.flush()
    fillTracks=[]
    for fg in fGs:
        for face in fg.faces:
            track = face.get_fillTrack()
            if track:
                fillTracks.append(track)
    with open("pickled_objs/"+set_name+"_fillTracks.obj", 'wb') as fp:
        cPickle.dump(fillTracks, fp)
        fp.close()
    print("OK")

    pool.close()
    pool.join()
    endTime = time.time()
    print()
    if not(loadImgs):
        print("Tiempo para procesar imgs '{}': {}".format(set_name,datetime.timedelta(seconds=imgsProcEnd-imgsProcStart)))
    print("Tiempo para procesar fGs (1): {}".format(datetime.timedelta(seconds=fgsProc1End-fgsProc1Start)))
    print("Tiempo total del script: {}".format(datetime.timedelta(seconds=endTime-startTime)))

