#For drawing Bezier Curve

import os
import yaml

import numpy as np
import random
from scipy.interpolate import barycentric_interpolate
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from Perlin_Noise import Perlin_Generator
import cv2

from Utils import Utils
utils = Utils()
normalize = utils.normalize
distance = utils.distance
scale_val = utils.scale_val


# ----------- Environment Parameters ----------- 

env_dimension = (600,400)

env_max_danger = 1
env_max_resource = 1
env_max_resource_gen_rate = 0.1

resource_consumption_rate = 1

danger_reduce = 1

max_gathered = 1


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
	def __init__(self, parameter_file):

		f = open(parameter_file)
		param = yaml.load(f, Loader = yaml.FullLoader)
		
		self.dimension_x, self.dimension_y = param['dimension_x'], param['dimension_y']

		print('Creating Environment with size ({},{})'.format(self.dimension_x, self.dimension_y))

		# Resource Generation Map
		self.resource_gen_rate = Perlin_Generator(dimension = (self.dimension_y, self.dimension_x), seed = 1).get_map()
		self.resource_gen_rate = normalize(self.resource_gen_rate, 0, env_max_resource_gen_rate)
		
		# Danger Map
		self.danger = Perlin_Generator(dimension = (self.dimension_y, self.dimension_x), seed = 2).get_map()
		self.danger = normalize(self.danger,0,env_max_danger)

		# Resource Map
		self.resource = Perlin_Generator(dimension = (self.dimension_y, self.dimension_x), seed = 3).get_map()
		self.resource = normalize(self.resource,0,env_max_resource)
		
		# Dynamic parameters of Agents
		self.max_social = 1
		self.max_inventory = 1
		self.max_reputation = 1
		self.max_damger = env_max_danger
		self.max_maslow = 1
		self.max_acq = 1

		self.max_resource = self.resource.sum()
	
	def update_parameters(self, sim):
		self.max_social = 0
		self.max_inventory = 0
		self.max_reputation = 0
		self.max_maslow = 0

		for agent_id, agent in sim.all_agents.items():
						
			if agent.social > self.max_social:
				self.max_social = agent.social

			if agent.inventory > self.max_inventory:
				self.max_inventory = agent.inventory

			if agent.reputation > self.max_reputation:
				self.max_reputation = agent.reputation

			if agent.maslow > self.max_maslow:
				self.max_maslow = agent.maslow

			for ag_id, score in agent.acquaint.items():
				if abs(score) > self.max_acq:
					self.max_acq = abs(score)

	
	def regen_resource(self):

		instant_rate = self.max_resource / self.resource.sum()
		self.resource += self.resource_gen_rate * instant_rate
		
		# Smoothening surface
		self.resource = cv2.GaussianBlur(self.resource,(3,3),0)
	

	def consume_resource(self, position):
		pos_x, pos_y = position

		# Argueable formula
		# Rule by which Agents extract reosurce from environment
		consumed = resource_consumption_rate * self.resource[pos_x,pos_y]

		if consumed > max_gathered:
			consumed = max_gathered

		self.resource[pos_x,pos_y] -= consumed
		self.resource = diffuse(self, self.resource, (pos_x, pos_y), radius = 2)

		# Agent's inventory gets updated
		return consumed
	

	def update_danger(self, sim):
		for agent_id, agent in sim.all_agents.items():
			
			pos_x, pos_y = agent.position

			# Argueable formula
			self.danger[pos_x, pos_y] -= danger_reduce * self.danger[pos_x, pos_y]
			self.danger = diffuse(self.danger, agent.position, radius = 3)


	def draw_agent(self, agent):

		pos_x, pos_y = agent.position
		plt.plot(pos_x,pos_y, marker = r'$\bigodot$', markersize = 35, color = agent.color) 
		
		agent_info = 'En: ' + str(int(agent.energy*10)) + '\n' + 'Inv: ' + str(int(agent.inventory * 10))
		
		plt.text(pos_x + 4 ,pos_y + 4, agent_info, fontsize=22)


	def draw_relationship(self, agent_src, agent_dst, score):

		# Relationships are shown with dotted lines.

		# Relationship line color (green - friend, red - foe)
		if score > 1:
			score = 1
		if score < -1:
			score  = -1

		if score > 0:
			b = 0
			r = 0
			g = 1
		else:
			b = 0
			r = 1
			g = 0

		# Relationship line transparency (intensity)
		if abs(score) > 0.3: 
			alph = abs(score) / 2

			src_x, src_y = agent_src.position 
			dst_x, dst_y = agent_dst.position

			plt.plot([dst_x, src_x], [dst_y, src_y], color = (r,g,b, alph), linestyle =':', linewidth=3)


	def draw_environment(self, sim, image_file = None, tick = None):
		
		plt.clf()

		plt.figure(figsize = (15,15), dpi=25)

		plt.rcParams["figure.figsize"] = [20,15]
		plt.axis('off')

		plt.title('Environment', fontsize = 22)
		

		plt.gca().invert_yaxis()

		plt.rc('xtick', labelsize=22)
		plt.rc('ytick', labelsize=22)

		res = plt.imshow(self.resource.T, cmap = 'Greens', alpha = 1)
		dng = plt.imshow(self.danger.T, cmap = 'Reds', alpha = 0.4)

		if tick is not None:
			plt.text(3 ,-5, 'Tick: {}'.format(tick), fontsize=18)

		plt.colorbar(res).set_label('Resource')
		plt.colorbar(dng).set_label('Danger')
		
		for agent_id, agent in sim.all_agents.items():
		
			self.draw_agent(agent)

			for other_agent_id, score in agent.acquaint.items():
				try:
					other_agent = sim.all_agents[other_agent_id]
					self.draw_relationship(agent, other_agent, score)
				except:
					pass

		if image_file:
			plt.savefig(image_file)

		plt.close()
	
		




	


