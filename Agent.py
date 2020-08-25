import numpy as np
import random
from scipy.interpolate import barycentric_interpolate
import matplotlib.pyplot as plt
import math

import tensorflow as tf


from Utils import Utils
from Agent_Brain import Brain

utils = Utils()
normalize = utils.normalize
distance = utils.distance
scale_val = utils.scale_val
roulette_selection = utils.roulette_selection
gradient = utils.gradient

total_no_agents = 0
invent_to_energy = 2
env_to_invent = 1
step = 1
time_unit_needs = 0.1
random_death_chance = 0
invent_fract = 0.5
energy_cons_rate = 0.3


phys_inputs = 3
phys_outputs = 5

class Eater:
	def __init__(self, sim, position):
		self.sim = sim
		self.position = position

class Sap:
	def __init__(self, sim, position,color = None):
		
		self.sim = sim
		self.ident = None

		# Physiological Attributes
		self.age = 0

		self.energy = 1
		self.social = 0.3
		self.actualization = 0.1

		self.maslow = 0.5

		# Physical Atttributes
		self.position = position
		self.inventory = np.random.uniform(0.3,0.6)


		#self.sight = int(math.sqrt(sim.dimension))
		self.sight = sim.dimension / 4

		# Social Attributes
		self.acquaint = {}
		self.color = list(np.random.uniform(size = 3)) if color is None else color
		
		self.memory = []
		self.max_memory = 3
		
		
		# Decisional brain
		'''
			1. Physiological decisions:

			states = [current_resource, inventory, energy]
			actions = [resource_priority, social_priority, explore_priority, consume_resource, gather_resource]

			- move_x, move_y are in range (-1,1) to compose any 360 movement
			- consume / gather resources in range (-1,1) and represents wether agent consumes 
				resource from the inventory to regenerate energy or gathers resource from the 
				environment into inventory

			Trained to maximize Maslow score and Energy
		

			2. Social decisions: 

			states = [energy, inventory, other-energy, other-inventory, other-maslow]
			actions = [fight / collaborate, mate]

			- fight / collaborate is in range (-1,1) and represents wether agents want to be adverse to the other or not
			- mate is a boolean and represents the reported by the agent compatibility of the two

			Trained to maximize Maslow score and Social
		'''
		#with tf.variable_creator_scope(self):
		self.phys_brain = Brain(no_inputs = phys_inputs, no_outputs = phys_outputs, architecture = (3,5))
		#self.social_brain = Brain(no_inputs = 5, no_outputs = 2, architecture = (6,10))

		self.phys_last_state = None
		self.phys_new_state = None
		self.phys_act = None

		'''
		self.soc_lasta_state = None
		self.soc_new_state = None
		self.soc_act = None
		'''

	def phys_decide(self):
		
		
		#Update current state
		pos_y, pos_x = self.position # Agent's position
		curr_resource = self.sim.env.resource[pos_x, pos_y] 

		self.phys_new_state = np.array([curr_resource, self.inventory, self.energy]).reshape(1, phys_inputs)
		
		if isinstance(self.phys_last_state, np.ndarray):
			
			self.phys_brain.remember(
				state = self.phys_last_state, 
				action = self.phys_act, 
				reward = -self.energy,
				new_state = self.phys_new_state
				)

			self.phys_brain.replay()
			self.phys_brain.target_train()

		self.phys_last_state = self.phys_new_state

		self.phys_act = self.phys_brain.decide(self.phys_new_state)

		action = self.phys_act

		print('Decision: {}'.format(action))

		if np.argmax(action) == 3:
			self.eat()

		elif np.argmax(action) == 4:
			self.gather_res()
		
		else:
			self.decide_direction(
				resource_priority = action[0],
				social_priority = action[1],
				explore_priority = action[2] )
		
	
	# Unit Time Run
	def life_tick(self):

		# Ageing
		self.age += 1 / 365

		# Attribute updates

		#self.update_inventory()
		self.update_maslow()
		self.update_social()
		self.compute_reputation()
		self.update_actualization()
		self.update_color()

		self.update_acquaint()
		self.propagate_acquaint()
		self.refresh_memory()

		self.phys_decide()

		# print('Ag', self.ident, ' acquaintances: ', self.acquaint.items())

		return self.stay_alive()

	def stay_alive(self):

		# Energy Consumption
		x, y = self.position
		self.energy -= time_unit_needs * self.sim.env.danger[x,y] * energy_cons_rate

		# Arbitrary death due to environment danger
		if self.energy <= 0 or np.random.uniform() < random_death_chance * self.sim.env.danger[x,y]:
			return 0 	# Dead
		else:
			return 1	# Alive


	def refresh_memory(self):
		if len(self.memory) > self.max_memory:
			memory.pop(0)


	def add_to_memory(self,agent_id):
		# Function that creates memory instance of internal state at the time of interaction
		self.memory.append((agent_id, self.maslow))


	def update_inventory(self):

		self.inventory = scale_val(self.inventory, self.sim.env.max_inventory, 1)
	

	def update_maslow(self, base = 2):

		#Arguable formula
		self.maslow = base ** 2 * self.energy + base * self.social + self.actualization

		self.maslow = scale_val(self.maslow, self.sim.env.max_maslow, 1)


	def update_social(self):
		# Only applies to agents known by self
		new_social = 0

		for agent_id in self.acquaint.keys():
			
			try:
				agent = self.sim.all_agents[agent_id]

				# Argueable formula
				# Social closeness - the closer to friendly agents the better
				dist = distance(self.position, agent.position)
				if dist < self.sight:
					new_social = (1 / (dist + 0.1)) * self.acquaint[agent.ident]
			except:
				pass

		self.social = (self.social + new_social ) / 2
		
		# Value validation
		if self.social < 0:
			self.social = 0

		# Rescale value to 0-1
		self.social = scale_val(self.social, self.sim.env.max_social,1)


	def compute_reputation(self):
		# Compute a reputation 
		rep = 0
		no = 0
	
		for agent_id, agent in self.sim.all_agents.items():
			if self.ident in agent.acquaint.keys():
				rep += agent.acquaint[self.ident]
				no += 1 

		if no:
			rep /= no

		return round(rep,2)


	def update_actualization(self):
		# Arguable formula
		self.actualization = (self.actualization + self.compute_reputation()) / 2

		# Has to include the (eventual) offspring maslow scores


	def vecinity(self):
		# Computes a score for the directions of movement

		self_x, self_y = self.position
		vecinity_x = 0
		vecinity_y = 0

		for agent_id, score in self.acquaint.items():
			try:
				agent = self.sim.all_agents[agent_id]
				ag_x, ag_y = agent.position
				# Only applies to agents known by self
		
				vecinity_x += 1 / (self_x - ag_x + 0.1) * score
				vecinity_y += 1 / (self_y - ag_y + 0.1) * score
			except:
				pass

		return vecinity_x, vecinity_y


	# Actions
	def decide_direction(self, resource_priority, social_priority, explore_priority):

		x, y = self.position

		# Compute resource <gradient>
		d_res_x, d_res_y = gradient(self.sim.env.resource, x, y)

		# Compute social <gradient>
		d_soc_x, d_soc_y = self.vecinity()

		# Exploration
		d_rand_x, d_rand_y = np.random.uniform(low = -1, high = 1, size = 2)

		dir_x = d_res_x * resource_priority + d_soc_x * social_priority + d_rand_x * explore_priority
		dir_y = d_res_y * resource_priority + d_soc_y * social_priority + d_rand_y * explore_priority

		self.move(dir_x, dir_y)


	def move(self, d_x, d_y):

		d_x = int(np.round(d_x))
		d_y = int(np.round(d_y))

		#print('dx: ', d_x, ', dy: ', d_y)
		
		if abs(d_x) > step:
			d_x = step * np.sign(d_x)

		if abs(d_y) > step:
			d_y = step * np.sign(d_y)

		pos_x, pos_y = self.position
		
		# Make sure agent stays in environment boundries
		if pos_x + d_x >= self.sim.env.dimension_x:
			pos_x = self.sim.env.dimension_x - 1 
		elif pos_x + d_x < 0:
			pos_x = 0
		else:
			pos_x += d_x

		if pos_y + d_y >= self.sim.env.dimension_y:
			pos_y = self.sim.env.dimension_y - 1
		elif pos_y + d_y < 0:
			pos_y = 0
		else:
			pos_y += d_y



		self.position = (pos_x, pos_y)
		self.energy -= math.sqrt(d_x ** 2 + d_y ** 2) * energy_cons_rate / 10
		

	def eat(self):
	
		
		self.energy += self.inventory / 2 * invent_to_energy
		self.inventory /= 2
		
		# Value validation
		if self.energy > 1:
			self.energy = 1

	def gather_res(self):
		# Arguable formula
	
		# Gathering resource is more efficient while near collaborating agents
		self.inventory += env_to_invent * self.sim.env.consume_resource(self.position)

		# Value validation
		if self.inventory < 0:
			self.inventory = 0

		self.inventory = scale_val(self.inventory, self.sim.env.max_inventory, 1)


	def learn_about(self, from_agent_id, about_agent_id, score):

		#print('Ag', self.ident, ' learned about ag', about_agent_id, ' from ag', from_agent_id, ' (', score, ')')
		if from_agent_id in self.acquaint.keys():
			if about_agent_id in self.acquaint.keys():
				self.acquaint[about_agent_id] += round(score * self.acquaint[from_agent_id],2)
			else:
				self.acquaint[about_agent_id] = round(score * self.acquaint[from_agent_id],2)


	def update_acquaint(self, base = 2):

		for mem in self.memory:

			agent_id, init_maslow = mem
			interact_score = (self.maslow - init_maslow) * base ** self.memory.index(mem)  

			if agent_id in self.acquaint.keys():
				new_acq_score = (self.acquaint[agent_id] + interact_score) / 2
			else:
				new_acq_score = interact_score


			self.acquaint[agent_id] = round(scale_val(new_acq_score, self.sim.env.max_acq, 1, -self.sim.env.max_acq, -1),2)


	def propagate_acquaint(self):

		# Propagate new interaction feedback to other agents
		# Other agents will learn about the agents' interaction and will update 
		#their own social scores accordingly to the reported interaction score and their score
		#with the agent communicating

		agent_id = roulette_selection(self.acquaint)
		
		if agent_id:
			score = self.acquaint[agent_id] / 10
			
			for other_agent_id in self.acquaint.keys():

				if other_agent_id != agent_id:

					other_agent = self.sim.all_agents[other_agent_id]
					other_agent.learn_about(self.ident, agent_id, score)


	def train(self,agent, iterations = 10):
		#To be implemented
		pass

	
	def update_color(self):
		# Updates color of agent as a means of social ignalling:
		# Cooperating agents will converge to simmilar colors
		for agent_id, score in self.acquaint.items():

			try:

				agent = self.sim.all_agents[agent_id]
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
			except:
				pass


	def social_policy(self, other_agent):

		#To be updated to an intelligent decision

		collaborate = self.acquaint[other_agent.ident] * np.random.uniform(-0.1,0.3)
		fight = - self.acquaint[other_agent.ident] * np.random.uniform(-0.1,0.3)
		mate = self.acquaint[other_agent.ident] * np.random.uniform(-0.1,0.1) * np.random.randint(0,2)

		return colaborate, fight, mate



class Sap_Interactions:

	def __init__(self, sim):
		# 

		self.sim = sim


	def pair_agent(self, src_agent, all_agents):


		options = {}

		options[-1] = max_interact_dist # Chance to select no agent to interact with

		for agent_id, agent in all_agents:
			dist = distance(src_agent, agent)
			if dist < src_agent.sight:
				options[agent.ident] = 1 / dist

		if bool(options):
			return roulette_selection(options)
		else:
			return -1


	def interact(self, agent1_id, agent2_id):
		c_1, f_1, m_1 = agent_1.social_policy(agent_2)
		c_2, f_2, m_2 = agent_2.social_policy(agent_1)

		if f_1 is max([c_1, f_1, m_1]) or f_2 is max([c_2, f_2, m_2]):
			self.fight(agent1_id, f_1, agent2_id, f_2)

		else:

			if m_1 is max([c_1, f_1, m_1]) and m_2 is max([c_2, f_2, m_2]):
				self.mate(agent1_id, agent2_id)

			else:
				self.collaborate(agent1_id, agent2_id)

		# Agents will remeber the interaction in order to decide if it wass positive or not
		agent1 = self.sim.all_agents[agent1_id]
		agent2 = self.sim.all_agents[agent2_id]

		agent1.add_to_memory(agent2_id)
		agent2.add_to_memory(agent1_id)


	def simulate_interactions(self):

		agents = dict(self.sim.all_agents).items()

		for agent_id, agent in agents:
			oth_agent_id = self.pair_agent(agent, agents)
			
			# Make sure same agent isn't paired with more than 1 agents
			del agents[agent_id]
			del agents[oth_agent_id]

			oth_agent = self.sim.all_agents[oth_agent_id]
			self.interact(agent, oth_agent)


	def fight(self, agent1_id, f1, agent2_id, f2):

		agent1 = self.sim.all_agents[agent1_id]
		agent2 = self.sim.all_agents[agent2_id]

		options = {}
		options[agent1_id] = f1 * agent1.energy
		options[agent2_id] = f2 * agent2.energy

		winner = roulette_selection(options)

		# Both agents lose energy 
		agent1.energy -= f1
		agent2.energy -= f2

		# Winner takes part of the loser's inventory
		def fight_result(winner, loser):
			winner.inventory += loser.inventory * invent_fract
			loser.inventory *= (1 - invent_fract)
		
		fight_result(winner, Diff([agent1, agent2],winner))


	def collaborate(self, agent1_id, agent2_id):

		agent1 = self.sim.all_agents[agent1_id]
		agent2 = self.sim.all_agents[agent2_id]

		if agent1.maslow > agent2.maslow:
			agent1.train(agent2)
			agent1.inventory += agent2.inventory * invent_fract
			agent2.inventory *= (1 - invent_fract)

		else:
			agent2.train(agent1)
			agent2.inventory += agent1.inventory * invent_fract
			agent1.inventory *= (1 - invent_fract)


	def mate(self, agent1_id, agent2_id):
		# 
		pass





