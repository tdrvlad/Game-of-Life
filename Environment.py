#For drawing Bezier Curve

import numpy as np
import random
from scipy.interpolate import barycentric_interpolate
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from Perlin_Noise import Perlin_Generator
import cv2


# ----------- Environment Parameters ----------- 

env_dimension = (600,400)

env_max_danger = 10
env_max_resource = 100
env_max_resource_gen_rate = 0.1

resource_consumption_rate = 1

patch_size = 50
patch_space = 10

danger_reduce = 0.1


def normalize(arr, new_min = 0,new_max = 1):
	if arr.min() < 0:
		arr -= arr.min()
	arr /= arr.max()

	arr *= (new_max - new_min)
	arr += new_min
	
	return arr

def diffuse(arr, pos, radius = 1):
	#Blur around a specific point of image
	pos_x, pos_y = pos

	diffuse_area = arr[pos_x - radius : pos_x + radius + 1, pos_y - radius : pos_y + radius + 1]
	diffuse_area = cv2.GaussianBlur(diffuse_area,(radius * 2 + 1,radius * 2 + 1),0)
	arr[pos_x - radius : pos_x + radius + 1, pos_y - radius : pos_y + radius + 1] = diffuse_area

	return arr

class Environment:
	def __init__(self,dimension):

		self.dimension_x, self.dimension_y = dimension

		#Compressed Map (for computational purposes)
		comp_r = 10
		comp_x, comp_y = (int(self.dimension_x / comp_r), int(self.dimension_y / comp_r))
		
		#Resource Generation Map
		#self.resource_gen_rate = Perlin_Generator(dimension = (comp_x, comp_y), seed = 1).get_map().repeat(comp_r, axis=0).repeat(comp_r, axis=1)
		self.resource_gen_rate = Perlin_Generator(dimension = (self.dimension_x, self.dimension_y), seed = 1).get_map()
		self.resource_gen_rate = normalize(self.resource_gen_rate,0,env_max_resource_gen_rate)
		
		#Danger Map
		#self.danger = Perlin_Generator(dimension = (comp_x, comp_y), seed = 2).get_map().repeat(comp_r, axis=0).repeat(comp_r, axis=1)
		self.danger = Perlin_Generator(dimension = (self.dimension_x, self.dimension_y), seed = 2).get_map()
		self.danger = normalize(self.danger,0,env_max_danger)

		#Resource Map
		self.resource = Perlin_Generator(dimension = (self.dimension_x, self.dimension_y), seed = 3).get_map()
		self.resource = normalize(self.resource,0,env_max_resource)
		
		#Agents in the Environment
		self.agents = []

	
	def get_resource(self):
		return self.resource

	def regen_resource(self):
		self.resource += np.random.uniform() * self.resource_gen_rate * (env_max_resource - self.resource)
		
		# Smoothening surface
		self.resource = cv2.GaussianBlur(self.resource,(5,5),0)
	

	def consume_resource(self,agent,consumed,pos):
		pos_x, pos_y = pos

		# Argueable formula
		# Rule by which Agents extract reosurce from environment
		consumed = resource_consumption_rate * self.resource[pos_x,pos_y]
		self.resource[pos_x,pos_y] -= consumed
		self.resource = diffuse(self.resource, pos, radius = 2)

		# Agent's inventory gets updated
		agent.inventory += consumed
		
	def update_danger(self):
		for agent in self.agents:
			pos_x, pos_y = agent.position

			# Argueable formula
			self.danger[pos_x, pos_y] -= danger_reduce * self.danger[pos_x, pos_y]
			self.danger = diffuse(self.danger, agent.position, radius = 2)

	def draw_environment(self):
		figure = plt.figure()
		plt.title('Environment')
		res = plt.imshow(self.resource, cmap = 'Greens', alpha = 0.8)
		dng = plt.imshow(self.danger, cmap = 'Reds', alpha = 0.5)
		
	
		plt.colorbar(res).set_label('Resource')
		plt.colorbar(dng).set_label('Danger')
		
		for agent in self.agents:
			pos_x, pos_y = agent.position
			plt.plot(pos_x,pos_y,marker = r'$\bigodot$', markersize = 15, color = agent.color) 

		plt.show()

if __name__ == '__main__':

env = Environment((200,200))

for i in range(20):
	x = np.random.uniform() * 200
	y = np.random.uniform() * 200
	env.agents.append(Agent((x,y)))

before = env.resource

env.draw_environment()


