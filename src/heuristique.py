import numpy as np

class SnailDirection(object):
    def __init__(self):
        self.d = 'R'
        self.pos = None
        self.MISSING_PIXEL = np.array([-100, -100, -100])
        self.failCpt = 0

    def switch(self):
        if self.d == 'R':
            self.d = 'D'
        elif self.d == 'D':
            self.d = 'L'
        elif self.d == 'L':
            self.d = 'U'
        elif self.d == 'U':
            self.d = 'R'

    def nextM(self, ind):
        i,j = ind
        if self.d == 'R':
            return  (i, j+1)
        elif self.d == 'D':
            return (i+1, j)
        elif self.d == 'L':
            return (i, j-1)
        elif self.d == 'U':
            return (i-1, j)

    def moveGenerator(self, pos, img, last):
        self.pos = pos
        yield pos
        while pos != last:
            if self.failCpt >= 4:
                raise StopIteration("bla")
            i,j = self.nextM(self.pos)
            try:
                v = img[i][j]
                print('img[i][j] :', v)
                outBound = False
            except:
                outBound = True
            if i < 0 or j < 0:
                outBound = True
            if (outBound or np.any(img[i][j] != self.MISSING_PIXEL)):
                print('switch')
                self.switch()
                print('new direction :', self.d)
                self.failCpt += 1
                
            else:
                self.pos = i,j
                yield self.pos
                self.failCpt = 0
                
            
    
