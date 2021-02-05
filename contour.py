import numpy as np
import cv2

class Contour(object):
    """description of class"""

    def __init__(self, con, imgHeight=1):
        x, y, w, h=cv2.boundingRect(con)
        
        self.X=x
        self.Y=y
        self.Width=w
        self.Height=h
        self.Center = (x+w/2, y+h/2)

        self.Ratio=h/float(w)
        self.Area=cv2.contourArea(con)
        self.RectArea=self.Width * self.Height
        self.PossibleLetter = self.CheckPossibility(imgHeight)


    def lieIn(self, Cont):
        left_corner = False
        right_corner = False

        if self.X <= Cont.X <= self.X + self.Width and self.Y <= Cont.Y <= self.Y + self.Height:
            left_corner = True
        
        if self.X <= Cont.X + Cont.Width <= self.X + self.Width and self.Y <= Cont.Y + Cont.Height <= self.Y + self.Height:
            right_corner = True

        if left_corner and right_corner:
            return True
        else:
            return False

    def CheckPossibility(self, imgH):
        if self.Ratio < 1.2:
           return False
        elif self.Ratio > 3.5:
           return False
        elif imgH/self.Height > 4:
           return False
        elif self.RectArea < 300:
           return False
        elif self.X == 0:
           return False
        elif self.Y == 0:
           return False
        else:
            return True