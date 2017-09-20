# -*- coding: utf-8 -*-
#!/usr/bin/python

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
        return {'fillLeft': self.fillLeft, 'fillTop': self.fillTop, 'fillRight': self.fillRight, 'fillBottom': self.fillBottom}
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
