# -*- coding: utf-8 -*-
#!/usr/bin/python

import dlib

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
