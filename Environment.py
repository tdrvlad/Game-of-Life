import numpy as np
import random
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt

def generate_points(min_x, max_x, min_y,max_y, no_points = 10):

	no_points = int(no_points)

	points_x = random.sample(range(min_x,max_x),no_points)
	points_y = random.sample(range(min_y,max_y),no_points)

	points = [(points_x[i], points_y[i]) for i in range(0, no_points)] 
    
	return points

def interpolate(points):

	x = []
	y = []
	for i in range(0,len(points)):
		x.append(points[i][0])
		y.append(points[i][1])

	return lagrange(x, y)

class Environment_Parameter:
	def __init__(self,min_x = 0, max_x = 10, min_y = 0, max_y = 10, no_units = 100):
		points = generate_points(min_x,max_x,min_y,max_y, no_points = no_units / 10)
		self.poly = interpolate(points)

		self.x_axis = np.linspace(min_x,max_x,no_units)
		self.y_axis = self.poly(self.x_axis)

	def show(self):

		plt.plot(self.x_axis, self.y_axis)
		plt.show()



	def grow(self, x_pos):
		pass

	def shrink(self, x_pos):
		pass


resources = Environment_Parameter()

resources.show()
