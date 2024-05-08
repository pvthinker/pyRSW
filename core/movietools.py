import subprocess
import os


class Movie(object):
    """ Home made class to generate mp4 """

    def __init__(self, fig, name='mymovie'):
        """ input: fig is the handle to the figure """
        self.fig = fig
        canvas_width, canvas_height = self.fig.canvas.get_width_height()
        # Open an ffmpeg process
        outf = '%s.mp4' % name
        self.outf = outf
        videoencoder = None
        for v in ['avconv', 'ffmpeg']:
            if subprocess.call(['which', v], stdout=subprocess.PIPE) == 0:
                videoencoder = v

        if videoencoder is None:
            print('\n')
            print('Neither avconv or ffmeg was found')
            print('Install one or set param.generate_mp4 = False')
            raise ValueError('Install avconv or ffmeg')

        cmdstring = (videoencoder,
                     '-y', '-r', '30',  # overwrite, 30fps
                     # size of image string
                     '-s', '%dx%d' % (canvas_width, canvas_height),
                     '-pix_fmt', 'argb',  # format
                     '-f', 'rawvideo',
                     # tell ffmpeg to expect raw video from the pipe
                     '-i', '-',
                     '-vcodec', 'libx264', outf)  # output encoding

        devnull = open(os.devnull, 'wb')
        self.process = subprocess.Popen(cmdstring,
                                        stdin=subprocess.PIPE,
                                        stdout=devnull,
                                        stderr=devnull)

    def addframe(self):
        string = self.fig.canvas.tostring_argb()
        self.process.stdin.write(string)

    def finalize(self):
        self.process.communicate()
