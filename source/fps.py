import time


class FPS(object):
    def __init__(self, name="unknown"):
        self.TotalTime = 0
        self.FrameTime = 0
        self.StartTime = 0
        self.TotalFrames = 0
        self.Name = name

    def start(self) -> None:
        self.TotalFrames += 1
        self.StartTime = time.time()

    def stop(self, print_FPS=False) -> None:
        self.FrameTime = time.time() - self.StartTime
        self.TotalTime += self.FrameTime
        if self.FrameTime != 0:
            fps = 1/self.FrameTime

        if print_FPS:
            print(self.Name+" FPS: ", fps)

    def logFps(self):
        if self.TotalTime != 0:
            fps = self.TotalFrames / self.TotalTime
        else:
            fps = 0
        log_msg = "{} : FPS = {:.2f}"

        return log_msg.format(self.Name, fps)


class FPSTracker(FPS):
    def __init__(self, *kwargs):
        self.FPSobjects = {}
        self.FPSobjects = {arg: FPS(arg) for arg in kwargs}

    def __setitem__(self, key, value=None):
        self.FPSobjects[key] = FPS(key)

    def __getitem__(self, item):
        return self.FPSobjects[item]

    def saveLog(self):
        with open('./log/Log_FPS.txt', 'w') as file:
            for fps_obj in self.FPSobjects.keys():
                file.write(self.FPSobjects[fps_obj].logFps()+"\n")
