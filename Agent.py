import numpy as np
import random
from scipy.interpolate import barycentric_interpolate
import matplotlib.pyplot as plt
import math

total_no_agents = 0
invent_to_needs = 2
step = 1

def distance(pos1,pos2):
	x1, y1 = pos1
	x2, y2 = pos2
	dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
	return dist  

def scale_val(val,old_max,new_max, old_min = 0, new_min = 0):
	return ((val - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min


class Sap:
	def __init__(self,position,color = None):
		
		global total_no_agents
		'''
		# Identity
		self.id = 
		self.color = 

		# Social
		self.acqint = {}
		self.family = {}

		# Physical Attributes
		self.position = position
		self.age
		self.inventory
		self.embodiment

		# Psychological Attributes
		self.needs
		self.social
		self.goods
		self.actualization
		'''

		self.ident = None
		self.color = list(np.random.uniform(size=3)) if color is None else color
		self.acquaint = {}
		self.family = {}
		self.position = position
		self.age = 0
		self.inventory = 0
		#self.embodiment = embodiment
		self.needs = 1
		self.social = 0.3
		self.actualization = 0.1

		#self.brain = tf.Sequential()

	# Unit Time Run
	def life_tick(self,all_agents, environment):

		self.age += 1 / 365

		self.update_maslow()
		self.update_social(all_agents, environment)
		self.compute_reputation(all_agents, environment)
		self.update_actualization(all_agents, environment)
		
		vec_x, vec_y = self.vecinity(all_agents, environment)
		#vec_x *= np.random.uniform(-1, 1)
		#vec_y *= np.random.uniform(-1, 1)

		self.move(vec_x, vec_y, environment)
		self.gather_res(environment)
		self.eat(1 / self.needs)

		agent_id, agent = random.choice(list(all_agents.items())) 
		self.update_acquaint(agent_id, np.random.uniform(-1,1))

		self.update_color(all_agents)

	
	# Decision Inputs
	
	def update_maslow(self, base = 2):

		#Arguable formula
		maslow = base ** 2 * self.needs + base * self.social + self.actualization

	def update_social(self, all_agents, environment):
		# Only applies to agents known by self
		new_social = 0

		for agent_id in self.acquaint.keys():
			agent = all_agents[agent_id]

			# Argueable formula
			# Social closeness - the closer to friendly agents the better
			dist = distance(self.position, agent.position)
			if dist != 0:
				new_social = (1 / dist) * self.acquaint[agent.ident]

		self.social = (self.social + new_social ) / 2
		
		# Value validation
		if self.social < 0:
			self.social = 0

		# Rescale value to 0-1
		self.social = scale_val(self.social,environment.max_social,1)

	def compute_reputation(self, all_agents, environemnt):
		# Compute a reputation 
		rep = 0
		no = 0
	
		for agent_id, agent in all_agents.items():
			if self.ident in agent.acquaint.keys():
				rep += agent.acquaint[self.ident]
				no += 1 

		if no:
			rep /= no

		return rep

	def update_actualization(self, all_agents, environment):
		# Arguable formula
		self.actualization = (self.actualization + self.compute_reputation(all_agents, environment)) / 2

	def vecinity(self, all_agents, environment):
		# Computes a score for the directions of movement

		pos_x, pos_y = self.position
		vecinity_x = 0
		vecinity_y = 0

		for agent_id, score in self.acquaint.items():
			agent = all_agents[agent_id]
			x, y = agent.position
			# Only applies to agents known by self
			
			# Argueable formula
			if x - pos_x != 0:
				vecinity_x += 1 / (x - pos_x) * score * environment.resource[x,y]
			if y - pos_y != 0:
				vecinity_y += 1 / (y - pos_y) * score * environment.resource[x,y]

		return vecinity_x, vecinity_y


	# Actions

	def move(self,x,y,environment):
		if x > step:
			x = step
		elif x < -step:
			x = -step
		else:
			pass

		if y > step:
			y = step
		elif y < -step:
			y = -step
		else:
			pass

		pos_x, pos_y = self.position
		
		# Make sure agent stays in environment boundries
		if pos_x + x >= environment.dimension_x:
			pos_x = environment.dimension_x - 1
		elif pos_x + x < 0:
			pos_x = 0
		else:
			pos_x += x

		if pos_y + y >= environment.dimension_y:
			pos_y = environment.dimension_y - 1
		elif pos_y + y < 0:
			pos_y = 0
		else:
			pos_y += y

		self.position = (int(pos_x), int(pos_y))
		
	def eat(self, hunger):
		quant = np.random.uniform(0.8,1.2) * hunger

		self.inventory -= quant
		self.needs += quant * invent_to_needs

		# Value validation
		if self.needs > 1:
			self.needs = 1

		if self.inventory < 0:
			self.inventory = 0

	def gather_res(self, environment):
		# Arguable formula
	
		self.inventory += environment.consume_resource(self.position) * self.social

		# Value validation
		if self.inventory < 0:
			self.inventory = 0

		self.inventory = scale_val(self.inventory,environment.max_inventory,1)

	def update_acquaint(self, agent_id, interact_score):
		if agent_id in self.acquaint.keys():
			self.acquaint[agent_id] = (self.acquaint[agent_id] + interact_score) / 2
		else:
			self.acquaint[agent_id] = interact_score

	def propagate_acquaint(self, agent_id, interact_score, all_agents):
		# Propagate new interaction feedback to other agents
		# Other agents will learn about the agents' interaction and will update 
		#their own social scores accordingly to the reported interaction score and their score
		#with the agent communicating

		for other_agent_id in self.acquaint.keys():
			if other_agent_id != agent_id:

				other_agent_score = self.acquaint[other_agent_id]

				other_agent = all_agents[other_agent_id]
				other_agent.update_acquaint(agent_id, other_agent_score * interact_score)

	def update_color(self, all_agents):
		# Updates color of agent as a means of social ignalling:
		# Cooperating agents will converge to simmilar colors
		for agent_id, score in self.acquaint.items():

			agent = all_agents[agent_id]
			oth_color = agent.color

			dist = distance(self.position,agent.position)
			if dist:
				score = score  / math.sqrt(dist)
			
				# Argueable formula
				color_delta = [oth_c - my_c for oth_c, my_c in zip(oth_color,self.color)]
				
				# Arguable formula
				#self.color = [my_c + c_delta * score / oth_c for my_c, c_delta, oth_c in zip(self.color, color_delta, agent.color)]
				self.color = [my_c + (c_delta * score / (my_c + oth_c + 0.01)) for my_c, c_delta, oth_c in zip(self.color, color_delta, oth_color)]

				self.color = [0 if c < 0 else c for c in self.color]
				self.color = [1 if c > 1 else c for c in self.color]





