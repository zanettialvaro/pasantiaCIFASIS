# -*- coding: utf-8 -*-
#!/usr/bin/python

import time
import cv2
import utils
import myRect
import myFace

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
        # print("Procesando imagen: '{}'".format(self.filepath))
        st = time.time()
        img = cv2.imread(self.filepath)
        #~ upscale = 0
        dets,scores,idx = utils.detector.run(img, upscale, utils.DETECTION_THR)
        if len(dets) == 0: #reintentar aplicando CLAHE
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_clahe = clahe.apply(gray)
            dets,scores,idx = utils.detector.run(img_clahe, upscale, utils.DETECTION_THR)
        if len(dets) > 0:
            for r,s,fID in zip(dets,scores,range(0,len(dets))):
                myR = myRect.myRect(r,s,False)
                lands = utils.face_pose_predictor(img, myR.get_rect())
                emb = utils.embedder.compute_face_descriptor(img, lands)
                face = myFace.myFace(myR,lands,emb,self)
                self.faces.append(face)
                self.nDetected += 1
        self.processTime = time.time() - st
