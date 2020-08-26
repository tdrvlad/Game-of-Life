import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

import sys
import time
import os
import glob
import yaml

from numpngw import write_apng
import gif

from Utils import Utils
from Perlin_Noise import Perlin_Generator

utils = Utils()
normalize = utils.normalize
distance = utils.distance
scale_val = utils.scale_val

parameter_file = 'Parameters.yaml'

# ----------- Environment Parameters ----------- 

param = yaml.load(open(parameter_file), Loader = yaml.FullLoader)

env_max_danger = param['env_max_danger']
env_max_resource = param['env_max_resource']
env_max_resource_gen_rate = param['env_max_resource_gen_rate']
danger_reduce = param['danger_reduce']
env_dimension = param['dimension_x'], param['dimension_y']

font_size = 40
fig_size = font_size

def diffuse(environment, arr, pos, radius = 1):
	#Blur around a specific point of image
	pos_x, pos_y = pos

	max_x, max_y = environment.dimension_x, environment.dimension_y

	if pos_x + radius < max_x and pos_x - radius > 0 and pos_y + radius < max_y and pos_y - radius > 0:

		diffuse_area = arr[pos_x - radius : pos_x + radius + 1, pos_y - radius : pos_y + radius + 1]
		diffuse_area = cv2.GaussianBlur(diffuse_area,(radius * 2 + 1,radius * 2 + 1), 0)
		arr[pos_x - radius : pos_x + radius + 1, pos_y - radius : pos_y + radius + 1] = diffuse_area

	return arr


class Environment:
	def __init__(self, parameter_file):

		self.dimension_x, self.dimension_y = env_dimension

		print('Creating Environment with size ({},{})'.format(self.dimension_x, self.dimension_y))

		# Resource Generation Map
		self.resource_gen_rate = Perlin_Generator(dimension = (self.dimension_x, self.dimension_y), seed = 1).get_map().T
		self.resource_gen_rate = normalize(self.resource_gen_rate, 0, env_max_resource_gen_rate)

		# Danger Map
		self.danger = Perlin_Generator(dimension = (self.dimension_x, self.dimension_y), seed = 2).get_map().T
		self.danger = normalize(self.danger,0,env_max_danger)

		# Resource Map
		self.resource = Perlin_Generator(dimension = (self.dimension_x, self.dimension_y), seed = 3).get_map().T
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
		consumed = self.resource[pos_x,pos_y] / env_max_resource * np.random.uniform(0.7, 1.3) 

		if consumed > self.resource[pos_x, pos_y]:
			consumed = self.resource[pos_x, pos_y]

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
		plt.plot(pos_x,pos_y, marker = r'$\bigodot$', markersize = font_size, color = agent.color) 
		
		#agent_info = 'En: ' + str(int(agent.energy*10)) + '\n' + 'Inv: ' + str(int(agent.inventory * 10))
		agent_info = str(int(self.resource[pos_x,pos_y] * 100))
		plt.text(pos_x + 3 ,pos_y + 3, agent_info, fontsize = 0.85 * font_size )


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
		plt.figure(figsize = (fig_size, fig_size), dpi=25)

		#plt.rcParams["figure.figsize"] = [40,40]
		#plt.axis('off')

		plt.title('Environment', fontsize = font_size)
		

		#plt.gca().invert_yaxis()

		plt.rc('xtick', labelsize = font_size)
		plt.rc('ytick', labelsize = font_size)

		res = plt.imshow(self.resource.T, cmap = 'Greens', alpha = 1)
		dng = plt.imshow(self.danger.T, cmap = 'Reds', alpha = 0)

		if tick is not None:
			plt.text(3 ,-5, 'Tick: {}'.format(tick), fontsize = font_size)

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
	
		




	


