#https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy

import numpy as np
import matplotlib.pyplot as plt

def perlin(x,y,seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u) # FIX1: I was using n10 instead of n01
    return lerp(x1,x2,v) # FIX2: I also had to reverse x1 and x2 here

def lerp(a,b,x):
    "linear interpolation"
    return a + x * (b-a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

class Perlin_Generator:
    def __init__(self, dimension = (100, 100), step = 5 , seed = 2):
        self.dimension_x, self.dimension_y = dimension
        self.step = step
        self.seed = seed

    def get_map(self):
        lin_x = np.linspace(0,self.step,self.dimension_x,endpoint=False)
        lin_y = np.linspace(0,self.step,self.dimension_y,endpoint=False)

        x,y = np.meshgrid(lin_x,lin_y) # FIX3: I thought I had to invert x and y here but it was a mistake

        return perlin(x,y,seed = self.seed)

if __name__ == '__main__':
    map = Perlin_Generator(dimension = (150,100), seed = 1).get_map().repeat(4, axis=0).repeat(4, axis=1)
    print(map)
    plt.imshow(map, origin = 'lower')
    plt.show()