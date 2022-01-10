import cv2


class FrameLoader(object):
    END_OF_FILE = True

    def __init__(self, path, start_frame=1, skip_frame=1):
        self.NextFrame = start_frame
        self.Capture = cv2.VideoCapture(path)
        self.TotalFrames = self.Capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.SkipFrames = skip_frame
        self.EmptyFrames = 0

        self.Capture.set(cv2.CAP_PROP_POS_FRAMES, self.NextFrame)

    def loader(self):
        while True:
            if self.Capture.isOpened():
                ret, undist = self.Capture.read()
            else:
                return self.END_OF_FILE

            if self._is_empty(ret):
                continue
            if self.NextFrame >= self.TotalFrames and not ret:
                return self.END_OF_FILE
            self.NextFrame += self.SkipFrames
            return undist

    def _is_empty(self, ret) -> bool:
        if self.NextFrame < int(self.TotalFrames) and not ret:
            self.EmptyFrames += 1
            self.NextFrame += self.SkipFrames
            self.Capture.set(cv2.CAP_PROP_POS_FRAMES, self.NextFrame)
            return True
        return False

