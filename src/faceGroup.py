# -*- coding: utf-8 -*-
#!/usr/bin/python

import utils
import myRect
import myFace
import fillTrack

from scipy import interpolate
import numpy as np
import os
import matplotlib
  
sshExec = False
if sshExec:
    matplotlib.use('Agg') #para usar por ssh
else:
    print("\nEn caso de ejecutar por ssh, poner en True sshExec para que no falle $DISPLAY env. var.\n")  

import matplotlib.pyplot as plt
import cv2
import dlib

splineFig = plt.figure(1,figsize=(20,10))

class faceGroup(object):
    def __init__(self,fgID):
        self.fgID = fgID
        self.faceID = None
        self.status = "No procesado (vacío)"
        self.faces = []
        self.imgsToInterpolate = []
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
        ol = utils.overlap(self.last_face.get_myRect().get_rect(),face.get_myRect().get_rect())
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

    def calculate_splines(self, set_name, plot = False):
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
            plt.suptitle("faceGroup id: {} - cant datos: {} - spline grad: {} - sPos = {} - sSizr = {}\npath = {}".format(self.fgID, len(t),k,sPos,sSize,fgPath))

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

            splineFig.savefig(os.path.join(utils.pathCheck(processed_path),"faceGropID{}.png".format(self.fgID)), bbox_inches='tight',dpi=300)
            splineFig.clf()
        #~ print('===========================================')

    def interpolate(self, save_interpolated = False):
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
            rect = utils.resize_negative_bounds(dlib.rectangle(l,t,r,b),img_height=cv2img.shape[0],img_width=cv2img.shape[1])
            myR = myRect.myRect(rect,-1,True)
            lands = utils.face_pose_predictor(cv2img, myR.get_rect())
            emb = utils.embedder.compute_face_descriptor(cv2img, lands)
            face = myFace.myFace(myR,lands,emb,imgti)
            face.set_id(self.faceID)
            self.faces.append(face)

            if save_interpolated:
                drawImg = copy.deepcopy(cv2img)
                draw_landmarks(drawImg,face.lands,None,red,thickness=3)
                cv2.imwrite(os.path.join(utils.pathCheck("./interpolatedImgs/fg{}".format(self.fgID)),os.path.basename(imgti.get_filepath())),drawImg)

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
                #~ print("h:{}-w:{} | t={} | fg{} - img:{}".format(img_height,img_width,t,self.fgID,face.get_inImage().get_filepath()))
                face.set_fillTop(-t)
                t = 0
            if b > img_height:
                #~ print("h:{}-w:{} | b={} | fg{} - img:{}".format(img_height,img_width,b,self.fgID,face.get_inImage().get_filepath()))
                face.set_fillBottom(b-img_height)
                b = img_height
            if l < 0:
                #~ print("h:{}-w:{} | l={} | fg{} - img:{}".format(img_height,img_width,l,self.fgID,face.get_inImage().get_filepath()))
                face.set_fillLeft(-l)
                l = 0
            if r > img_width:
                #~ print("h:{}-w:{} | r={} | fg{} - img:{}".format(img_height,img_width,r,self.fgID,face.get_inImage().get_filepath()))
                face.set_fillRight(r-img_width)
                r = img_width

            rect = dlib.rectangle(l,t,r,b)
            sRect = myRect.myRect(rect,-2,False)
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
            if r-l < 128 or b-t < 128:
                #~ print("\n\t|save_smoothed_to_disk(): (fgID: {}) {} (faceID: {} - interpolated: {}) muy pequeña (widht = {} - height = {}), no se guardará.".format(self.fgID,os.path.basename(path),face.get_id(),face.is_interpolated(),r-l,b-t))
                face.set_tooSmall()
                continue

            cv2img = cv2.imread(path)
            newPath = path.replace("test_sets",cropped_dest,1) #el primer argumento debe ser la carpeta donde están contenidos los datasets, de manera de guardar los resultados en un path similar donde se reemplaza test_sets por 'cropped_dest', se generará una carpeta 'cropped_dest' con las respectivas subcarpetas.
            folder,basename = os.path.split(newPath)
            if folder_structure:
                facePath = os.path.join(utils.pathCheck(os.path.join(folder,"fgID{}".format(self.fgID))),basename)
            else:
                if split_relevant:
                    if face.get_relevant():
                        classFolder = "relevantID/"
                    else:
                        classFolder = "notRelevantID/"
                else:
                    classFolder = ""
                facePath = os.path.join(utils.pathCheck(os.path.join(os.path.join(os.path.join("./",cropped_dest),classFolder),"fgID{}".format(self.fgID))),basename)

            if draw_lands:
                cen = face.lands.part(27)
                p33 = face.lands.part(33)
                cv2.circle(cv2img,(cen.x,cen.y),2,(0,255,0),2)
                cv2.circle(cv2img,(p33.x,p33.y),2,(0,0,255),2)
                if face.is_interpolated():
                    draw_landmarks(cv2img,face.lands,None,blue,thickness=2)
                else:
                    draw_landmarks(cv2img,face.lands,None,white,thickness=2)

            fills = face.get_fills()
            fillL = fills['fillLeft']
            fillT = fills['fillTop']
            fillR = fills['fillRight']
            fillB = fills['fillBottom']
            
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
                track = fillTrack.fillTrack(self.fgID,fills,facePath,origShape)
                face.set_fillTrack(track)

            if np.abs(croppedImg.shape[0] - croppedImg.shape[1]) > 1:
                print("\nsave_smoothed_to_disk(): fgID: {} - croppedImg: {} | origShape: {} - newShape: {}".format(self.fgID,os.path.basename(facePath),origShape,croppedImg.shape))
            hMax = 256
            wMax = 256
            #~ ratio = min(wMax/float(wIm), hMax/float(hIm))
            #~ wNew,hNew = int(wIm*ratio),int(hIm*ratio)
            resizedImg = cv2.resize(croppedImg,(wMax,hMax),interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(facePath,resizedImg)
