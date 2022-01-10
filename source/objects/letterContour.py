import numpy as np
import cv2


class Contour:
    """Class that represent contour of the letter - store all info about it (Position on img, size, area, etc.)"""

    def __init__(self, con=np.ones(0), img_height=1):
        # If no values are submitted with declaration __init__ creates empty object with default parameters.
        # empty contour object is created at the beginning of plate detection

        if con.shape[0] == 0:
            x, y, w, h = [1, 1, 1, 1]
            self.Area = 0
        else:
            x, y, w, h = cv2.boundingRect(con)
            self.Area = cv2.contourArea(con)

        self.X = x
        self.Y = y
        self.Width = w
        self.Height = h
        self.Center = (x+w/2, y+h/2)
        self.Ratio = h/float(w)

        self.RectArea = self.Width * self.Height
        self.PossibleLetter = self.CheckPossibility(img_height)

    def lieIn(self, Cont):
        # This method compere if one contour leis in another
        left_corner = False
        right_corner = False

        if self.X <= Cont.X <= self.X + self.Width and \
                self.Y <= Cont.Y <= self.Y + self.Height:
            left_corner = True

        if self.X <= Cont.X + Cont.Width <= self.X + self.Width and\
                self.Y <= Cont.Y + Cont.Height <= self.Y + self.Height:
            right_corner = True

        if left_corner and right_corner:
            return True
        else:
            return False

    def CheckPossibility(self, imgH):
        # Check if all criteria for letter possibility are met
        if self.Ratio < 1.3:
           return False
        elif self.Ratio > 3.9:
           return False
        elif imgH/self.Height > 4:
           return False
        elif self.RectArea < 500:
           return False
        elif self.X == 0:
           return False
        elif self.Y == 0:
           return False
        else:
            return True
