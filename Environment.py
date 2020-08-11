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
env_max_resource_gen_rate = 0.2

resource_consumption_rate = 2

danger_reduce = 1

def scale_val(val,old_max,new_max, old_min = 0, new_min = 0):
	return ((val - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min

def scale_arr(arr,new_max, new_min = 0):
	old_max = arr.max()
	old_min = arr.min()
	return ((arr - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min

def normalize(arr, new_min = 0, new_max = 1):
	if arr.min() < 0:
		arr -= arr.min()
	arr /= arr.max()

	arr *= (new_max - new_min)
	arr += new_min
	
	return arr

def diffuse(environment, arr, pos, radius = 1):
	#Blur around a specific point of image
	pos_x, pos_y = pos

	max_x, max_y = environment.dimension_x, environment.dimension_y

	if pos_x + radius < max_x and pos_x - radius > 0 and pos_y + radius < max_y and pos_y - radius > 0:

		diffuse_area = arr[pos_x - radius : pos_x + radius + 1, pos_y - radius : pos_y + radius + 1]
		diffuse_area = cv2.GaussianBlur(diffuse_area,(radius * 2 + 1,radius * 2 + 1),0)
		arr[pos_x - radius : pos_x + radius + 1, pos_y - radius : pos_y + radius + 1] = diffuse_area

	return arr

class Environment:
	def __init__(self,dimension):

		self.dimension_x, self.dimension_y = dimension

			
		# Resource Generation Map
		#self.resource_gen_rate = Perlin_Generator(dimension = (comp_x, comp_y), seed = 1).get_map().repeat(comp_r, axis=0).repeat(comp_r, axis=1)
		self.resource_gen_rate = Perlin_Generator(dimension = (self.dimension_x, self.dimension_y), seed = 1).get_map()
		self.resource_gen_rate = normalize(self.resource_gen_rate,0,env_max_resource_gen_rate)
		
		# Danger Map
		#self.danger = Perlin_Generator(dimension = (comp_x, comp_y), seed = 2).get_map().repeat(comp_r, axis=0).repeat(comp_r, axis=1)
		self.danger = Perlin_Generator(dimension = (self.dimension_x, self.dimension_y), seed = 2).get_map()
		self.danger = normalize(self.danger,0,env_max_danger)

		# Resource Map
		self.resource = Perlin_Generator(dimension = (self.dimension_x, self.dimension_y), seed = 3).get_map()
		self.resource = normalize(self.resource,0,env_max_resource)
		
		# Dynamic parameters of Agents
		self.max_social = 1
		self.max_inventory = 1
		self.max_reputation = 1

		self.max_resource = self.resource.sum()
	
	def update_parameters(self, all_agents):
		self.max_social = 0
		self.max_inventory = 0
		self.max_reputation = 0

		for agent_id, agent in all_agents.items():
						
			if agent.social > self.max_social:
				self.max_social = agent.social

			if agent.inventory > self.max_inventory:
				self.max_inventory = agent.inventory

			if agent.reputation > self.reputation:
				self.max_reputation = agent.max_reputation
	
	def regen_resource(self):

		instant_rate = self.max_resource / self.resource.sum()
		self.resource += self.resource_gen_rate * instant_rate
		
		# Smoothening surface
		self.resource = cv2.GaussianBlur(self.resource,(3,3),0)
	

	def consume_resource(self,position):
		pos_x, pos_y = position

		# Argueable formula
		# Rule by which Agents extract reosurce from environment
		consumed = resource_consumption_rate * self.resource[pos_x,pos_y]
		self.resource[pos_x,pos_y] -= consumed
		self.resource = diffuse(self, self.resource, (pos_x, pos_y), radius = 3)

		# Agent's inventory gets updated
		return consumed
		
	def update_danger(self, all_agents):
		for agent_id, agent in all_agents.items():
			
			pos_x, pos_y = agent.position

			# Argueable formula
			self.danger[pos_x, pos_y] -= danger_reduce * self.danger[pos_x, pos_y]
			self.danger = diffuse(self.danger, agent.position, radius = 2)

	def draw_environment(self, all_agents, image_file = None):
		
		plt.clf()

		plt.figure()

		plt.rcParams["figure.figsize"] = [20,15]
		plt.axis('off')

		plt.title('Environment')

		res = plt.imshow(self.resource, cmap = 'Greens', alpha = 1)
		dng = plt.imshow(self.danger, cmap = 'Reds', alpha = 0.3)
		
		
		plt.colorbar(res).set_label('Resource')
		plt.colorbar(dng).set_label('Danger')
		
		positions = []
		for agent_id, agent in all_agents.items():
		

			pos_x, pos_y = agent.position
			if (pos_x, pos_y) in positions:
				pos_x += 0.5
				pos_y += 0.5
			positions.append((pos_x,pos_y))

			agents = plt.plot(pos_y,pos_x, marker = r'$\bigodot$', markersize = 35, color = agent.color) 
		
		if image_file:
			plt.savefig(image_file)

		return plt
		




	


