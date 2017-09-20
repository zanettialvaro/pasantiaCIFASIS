# -*- coding: utf-8 -*-
#!/usr/bin/python

class fillTrack(object):
    def __init__(self,fg,fills,facePath,origShape):
        self.fg = fg
        self.fillLeft = fills['fillLeft']
        self.fillTop = fills['fillTop']
        self.fillRight = fills['fillRight']
        self.fillBottom = fills['fillBottom']
        self.facePath = facePath
        self.origShape = origShape
        self.resShape = (origShape[0]+self.fillTop+self.fillBottom,origShape[1]+self.fillLeft+self.fillRight,origShape[2])
        #~ print("fg: {} - croppedImg: {} | origShape: {} - resShape: {}".format(self.fg,self.facePath,self.origShape,self.resShape))

    def get_fg(self):
        return self.fg
    def get_fills(self):
        return {'fillLeft': self.fillLeft, 'fillTop': self.fillTop, 'fillRight': self.fillRight, 'fillBottom': self.fillBottom}
    def get_facePath(self):
        return self.facePath
    def get_origShape(self):
        return self.origShape
    def get_resShape(self):
        return self.resShape
