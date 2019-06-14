import os
import subprocess
import time
from inspect import isclass

class TimerBlock:
    def __init__(self, title):
        print(("{}".format(title)))

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.clock()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")

    def log(self, string):
        duration = time.clock() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print("  [{:.3f}{}] {}".format(duration, units, string), flush = True)

def module_to_dict(module, exclude=[]):
    return dict([(x, getattr(module, x)) for x in dir(module)
                 if isclass(getattr(module, x))
                 and x not in exclude
                 and getattr(module, x) not in exclude])


# creat_pipe: adapted from https://stackoverflow.com/questions/23709893/popen-write-operation-on-closed-file-images-to-video-using-ffmpeg/23709937#23709937
# start an ffmpeg pipe for creating RGB8 for color images or FFV1 for depth
# NOTE: this is REALLY lossy and not optimal for HDR data. when it comes time to train
# on HDR data, you'll need to figure out the way to save to pix_fmt=rgb48 or something
# similar
def create_pipe(pipe_filename, width, height, frame_rate=60, quite=True):
    # default extension and tonemapper
    pix_fmt = 'rgb24'
    out_fmt = 'yuv420p'
    codec = 'h264'

    command = ['ffmpeg',
               '-threads', '2',  # number of threads to start
               '-y',  # (optional) overwrite output file if it exists
               '-f', 'rawvideo',  # input format
               '-vcodec', 'rawvideo',  # input codec
               '-s', str(width) + 'x' + str(height),  # size of one frame
               '-pix_fmt', pix_fmt,  # input pixel format
               '-r', str(frame_rate),  # frames per second
               '-i', '-',  # The imput comes from a pipe
               '-an',  # Tells FFMPEG not to expect any audio
               '-codec:v', codec,  # output codec
               '-crf', '18',
               # compression quality for h264 (maybe h265 too?) - http://slhck.info/video/2017/02/24/crf-guide.html
               # '-compression_level', '10', # compression level for libjpeg if doing lossy depth
               '-strict', '-2',  # experimental 16 bit support nessesary for gray16le
               '-pix_fmt', out_fmt,  # output pixel format
               '-s', str(width) + 'x' + str(height),  # output size
               pipe_filename]
    cmd = ' '.join(command)
    if not quite:
        print('openning a pip ....\n' + cmd + '\n')

    # open the pipe, and ignore stdout and stderr output
    DEVNULL = open(os.devnull, 'wb')
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=DEVNULL, stderr=DEVNULL, close_fds=True)

# AverageMeter: code from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)