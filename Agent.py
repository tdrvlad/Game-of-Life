import numpy as np
import random
from scipy.interpolate import barycentric_interpolate
import matplotlib.pyplot as plt

distance(pos1,pos2):
	x1, y1 = pos1
	x2, y2 = pos2
	dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
	return dist  



class Sap:
	def __init__(self,position,color = None):
		self.position = position
		self.x, self.y = self.position

		self.needs 
		self.comfort 
		self.actualization

		self.inventory = 0
		self.age = 0
		self.offspring = []



		if color != None:
			self.color = color
		else:
			self.color = list(np.random.choice(range(256), size=3) / 255)
	
	def update_neighbours(self):
		pass
	
	def update_color(self)

		for color, score in self.neighbours_list:
			# Argueable formula
			self.color = (color - self.color) * score

	def vecinity(self, environment)
		vecinity_x = 0
		vecinity_y = 0
		for agent in self.neighbours_list:
			x, y = agent.position

			# Argueable formula
			vecinity_x += 1 / (x - self.x) * score(self,agent) * environment.resource[x,y]
			vecinity_y += 1 / (y - self.y) * score(self,agent) * environment.resource[x,y]

		return vecinity_x, vecinity_y



