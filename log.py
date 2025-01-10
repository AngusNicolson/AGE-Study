
import os


class Logger:
    def __init__(self, log_filename, display=True):
        self.log_filename = log_filename
        self.display = display
        self.f = open(log_filename, 'a')
        self.counter = [0]

    def __call__(self, text):
        if self.display:
            print(text)
        self.f.write(text + '\n')
        self.counter[0] += 1
        if self.counter[0] % 10 == 0:
            self.f.flush()
            os.fsync(self.f.fileno())

    def close(self):
        self.f.close()
