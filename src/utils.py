# -*- coding: utf-8 -*-
#!/usr/bin/python

from __future__ import print_function
import re
import locale
import os
import dlib
import cv2
import myFace
import myRect
#puede faltar algún import, se acomodó el código al final y no se probó toda la funcionalidad.

DETECTION_THR = 0

detector = dlib.get_frontal_face_detector()
predictor_model = "shape_predictor_68_face_landmarks.dat"
face_pose_predictor = dlib.shape_predictor(predictor_model)
rec_model_v1 = "dlib_face_recognition_resnet_model_v1.dat"
embedder = dlib.face_recognition_model_v1(rec_model_v1)

def flattenList(l):
    return [item for sublist in l for item in sublist]

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

def draw_rectangle(img, myR, color=(255,255,255), thickness=2):
    cv2.rectangle(img,(myR.get_rect().left(),myR.get_rect().top()),(myR.get_rect().right(),myR.get_rect().bottom()),color,thickness)

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

def show_img_with_rect(face):
    img = cv2.imread(face.get_inImage().get_filepath())
    myR = face.get_myRect()
    draw_rectangle(img, myR, color=(255,255,255), thickness=2)
    cv2.imshow('',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

def save_face(face,savePath,height,width):
    imgPath = face.get_inImage().get_filepath()
    print("\nguardando face: {}".format(imgPath))
    facePath = os.path.join(pathCheck(savePath),os.path.basename(imgPath))
    img = cv2.imread(imgPath)
    myR = face.get_myRect()
    rect = myR.get_rect()
    rect = resize_negative_bounds(rect, img_height=height, img_width=width)
    l,t,r,b = rect.left(), rect.top(), rect.right(), rect.bottom()
    hMax = 256
    wMax = 256
    if r-l < 128 or b-t < 128:
        print("save_face(): no se guardará {} por ser muy pequeña".format(os.path.basename(facePath)))
    else:
        croppedImg = img[t:b+1, l:r+1]
        resizedImg = cv2.resize(croppedImg,(wMax,hMax),interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(facePath,resizedImg)

def resize_negative_bounds(rect, img_height=800, img_width=1920):
    l,t,r,b = rect.left(),rect.top(),rect.right(),rect.bottom()
    l = max(0,l)
    t = max(0,t)
    r = min(r,img_width)
    b = min(b,img_height)
    return dlib.rectangle(l,t,r,b)

#-----------------------------------------------------------------------

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
