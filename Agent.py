import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

import sys
import time
import os
import glob
import yaml

from numpngw import write_apng
import gif

from Utils import Utils, Logger, Genetics
from Agent_Brain import Brain

# ----------- Tools ----------- 

utils = Utils()
normalize = utils.normalize
distance = utils.distance
scale_val = utils.scale_val
roulette_selection = utils.roulette_selection
gradient = utils.gradient
one_hot = utils.one_hot

parameter_file = 'Parameters.yaml'
log_file = 'Sim_Log.txt'
l = Logger(log_file)

genetic = Genetics()

# ----------- Agent Parameters ----------- 

param = yaml.load(open(parameter_file), Loader = yaml.FullLoader)
res_to_invent = param['res_to_invent']
invent_to_energy = param['invent_to_energy']
invent_fract = param['invent_fract']

ag_basic_needs = param['ag_basic_needs']
ag_move_needs = param['ag_move_needs']
ag_random_death = param['ag_random_death']

ag_max_sight = param['ag_sight']
ag_max_step = param['ag_max_step']
ag_max_invent = param['ag_max_invent']
ag_max_interact_dist = param['ag_max_interact_dist']
ag_mating_score = param['ag_mating_score']

colab_bonus = param['colab_bonus']
ag_blood_relation_thresh = param['ag_blood_relation_thresh']
ag_compatibility_thresh = param['ag_compatibility_thresh']

brain_inputs = 4
brain_outputs = 5

def agent_info(agent):
	info = 'Sim: ' + str(agent.sim.time) \
		+ ' | Invent:' + str(np.round(agent.inventory,1)) \
		+ ' | Energy:' + str(np.round(agent.energy,1)) + ' | Soc:' +  str(np.round(agent.social,1)) \
		+ ' | Act:' +  str(np.round(agent.actualization,1)) \
		+ ' | Maslw:' + str(np.round(agent.actualization,1))
	return info	

def decision_info(agent):
	info = '[Res: ' + str(np.round(agent.act[0],1)) \
	+ ' Soc: ' + str(np.round(agent.act[1],1)) \
	+ ' Expl: ' + str(np.round(agent.act[2],1)) \
	+ ' Fight: ' + str(np.round(agent.act[3],1)) \
	+ ' Collab: ' + str(np.round(agent.act[4],1)) \
	+ ']'

	return info
		

# ----------- Agent Class ----------- 

class Sap:
	
	# ------- Elementary ------- 

	def __init__(self, sim, position, parent1 = None, parent2 = None):
		
		self.sim = sim
		self.ident = None

		# --- Genetics --- 
		# Parameters that an agents inherits from parents

		self.color = genetic.get_color(parent1, parent2)
		self.dna = genetic.get_dna(parent1, parent2, brain_inputs, brain_outputs) # Architecture of the Neural Network

		# --- Physiological Attributes --- 
		# Explained in Agent definition

		self.age = 0

		self.energy = 1
		self.social = 0.3
		self.reputation = 0
		
		self.maslow = 0.5
		self.actualization = 0.1

		# --- Physical Atttributes --- 

		self.position = position
		self.inventory = np.random.uniform(0.3,0.6)
		self.max_invent = ag_max_invent
		self.sight = ag_max_sight
		self.shape = 1

		# Current movement step
		self.d_x = np.random.uniform(-1,1)
		self.d_y = np.random.uniform(-1,1)

		# --- Social Attributes --- 

		self.acquaint = {}
		self.family = {}
		self.offsprings = []
	
		self.memory = []
		self.max_memory = 3

		# No of agents the agent is collaborating with at a time unit
		self.colab = 0
		
		# --- Agent's decisional functions --- 

		self.brain = Brain(no_inputs = brain_inputs, no_outputs = brain_outputs, architecture = self.dna)
		
		# Needed for comparing rewards
		self.last_state = None
		self.new_state = None
		self.act = None

			
	def life_tick(self):
		# Unit Time Run
		
		# Ageing
		self.age += 1 / 365

		# Attribute updates
		self.update_maslow()
		self.update_social()
		self.update_reputation()
		self.update_actualization()
		self.update_color()

		self.update_acquaint()
		self.propagate_acquaint()
		self.refresh_memory()

		self.decide()

		return self.stay_alive()


	def stay_alive(self):
		# Update of vital parameters

		# Energy Consumption
		x, y = self.position
		self.energy -= ag_basic_needs * self.sim.env.danger[x,y]

		if self.energy < 0.1:
			self.energy = 0.1
		# Eating or gathering resources
		if 1 / self.energy > 1 / self.inventory:
			self.eat()
		else:
			self.gather_res()

		# Update of inventory
		if self.inventory < 0:
			self.inventory = 0

		if self.inventory > self.max_invent:
			self.inventory = self.max_invent

		self.log.log(agent_info(self))

		if self.energy < 0.1 or np.random.uniform() * self.age < ag_random_death * self.sim.env.danger[x,y]:
			return 0 	# Dead
		else:
			return 1	# Alive


	# ------- Physiological decisions and actions ------- 

	def decide(self, other_agent = None):

		# Brain
		'''
			inputs = [current_resource, inventory, energy, acquintance_score]
			actions = [resource_priority, social_priority, explore_priority, fight, collaborate]

			Agent decides wether to move or to interact with surrounding agents.
			If it chooses to move, the x_priority modulates a corresponding direction tensor (oriented by the parameter gradient)
			If it chooses to interact, policy is established by the maximum of the 3 - fight/collaborate
			If one agent chooses to interact and the other to move, the interaction is neutral
			If one agent chooses to fight, then interaction is fhigt and a winner is determined 
			If both choose to collaborate, then interaction is collaborate. If both rate collaboartion with high scores, they mate

			Brain is a Neural-Network trained to maximize Maslow Score
			Architecture of the Neural-Network is determined genetically

		'''

		#Update current state
		pos_x, pos_y = self.position # Agent's position
		curr_resource = self.sim.env.resource[pos_x, pos_y] 

		# Exception case: Interacting with an un-acquainted agent or modulating acquintance
		if other_agent is None:
			acq_score = 0.5
		else:
			if other_agent.ident not in self.acquaint.keys():
				acq_score = 0.5
			else:
				acq_score = (self.acquaint[other_agent.ident] + 1) / 2 # normalized with middle in 0.5
		
		self.new_state = np.array([curr_resource, self.inventory, self.energy, acq_score]).reshape(1, brain_inputs)
		

		if isinstance(self.last_state, np.ndarray):
			
			self.brain.remember(
				state = self.last_state, 
				action = self.act, 
				reward = self.maslow,
				new_state = self.new_state
				)

			self.brain.replay(epochs = 5)
			self.brain.train()

		self.last_state = self.new_state

		#print('Ag{} action: {}'.format(self.ident, self.act))
		
		self.act = np.round(self.brain.decide(self.new_state),2)

		self.log.log('Decision: ' + decision_info(self)) 

	
		if other_agent is None: # Non social decision

			self.direction(
				resource_priority = self.act[0],
				social_priority = self.act[1],
				explore_priority = self.act[2] )

			return None, None
		
		else:
			# Decide if social interaction is priority
			if (self.act[0] + self.act[1] + self.act[2]) / 3 > (self.act[3] + self.act[4]) / 2:
				self.direction(
					resource_priority = self.act[0],
					social_priority = self.act[1],
					explore_priority = self.act[2] )
				return None, None
			else:
				return self.act[3], self.act[4]

	
	def direction(self, resource_priority, social_priority, explore_priority):

		x, y = self.position

		# Retain only the maximum the priorities
		new_dir = np.round(one_hot(np.array([resource_priority, social_priority, explore_priority])),2)
		new_dir_factor = new_dir.max()


		# Compute resource <gradient>
		d_res_x, d_res_y = np.round(gradient(self.sim.env.resource, x, y),2)

		# Compute social <gradient>
		d_soc_x, d_soc_y = np.round(self.vecinity(),2)

		# Exploration <gradient>
		d_expl_x, d_expl_y = np.round(np.random.uniform(low = -1, high = 1, size = 2),2)

		self.d_x = np.round(self.d_x * ( 1 - new_dir_factor) + new_dir.T.dot(np.array([d_res_x, d_soc_x, d_expl_x])) * ag_max_step, 2)
		self.d_y = np.round(self.d_y * ( 1 - new_dir_factor) + new_dir.T.dot(np.array([d_res_y, d_soc_y, d_expl_y])) * ag_max_step, 2)
		
		'''
		print('MOVE({},{}) - Res:{}*({},{}), Soc:{}*({},{}), Expl:{}*({},{})' \
			.format(self.d_x, self.d_y, resource_priority, d_res_x, d_res_y, social_priority, \
			d_soc_x, d_soc_y, explore_priority, d_expl_x, d_expl_y), flush = True)
		'''
		
		self.move()


	def vecinity(self):
		# Computes direction tensor for the friendly agents
		# [i,j] pair that is oriented towards agents proportionaly to their acquintance score

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

		try:

			maxim = max(abs(vecinity_x), abs(vecinity_y))
			
			if maxim > 0:
				vecinity_x /= maxim
				vecinity_y /= maxim

			vecinity_x = int(np.round(vecinity_x))
			vecinity_y = int(np.round(vecinity_y))

		except:
			vecinity_x = 0
			vecinity_y = 0

		return vecinity_x, vecinity_y


	def move(self):

		self.d_x *= np.random.uniform(0.95,1.05)
		self.d_y *= np.random.uniform(0.95,1.05)

		if abs(self.d_x) > ag_max_step:
			self.d_x = ag_max_step * np.sign(self.d_x)

		if abs(self.d_y) > ag_max_step:
			self.d_y = ag_max_step * np.sign(self.d_y)

		
		d_x = self.d_x * (self.energy / ag_move_needs)
		d_y = self.d_y * (self.energy / ag_move_needs)

		d_x = int(np.round(d_x))
		d_y = int(np.round(d_y))

			
		pos_x, pos_y = self.position
		
		# Make sure agent stays in environment boundries
		if pos_x + d_x >= self.sim.env.dimension_x:
			d_x *= -1
		elif pos_x + d_x < 0:
			d_x *= -1
		
		pos_x += d_x

		if pos_y + d_y >= self.sim.env.dimension_y:
			d_y *= -1
		elif pos_y + d_y < 0:
			d_y *= -1
		
		pos_y += d_y


		traveled = math.sqrt(d_x ** 2 + d_y ** 2)

		self.position = (pos_x, pos_y)
		self.energy -= traveled * ( ag_move_needs / (colab_bonus * self.colab + 1))
		
		if self.energy < 0:
			self.energy = 0
			self.eat()
	
		# Note: Collaboration makes travelling demand less resource



	def eat(self):
	
		if self.inventory < 0:
			self.inventory = 0

		self.energy += self.inventory / 2 * invent_to_energy
		
		# Value validation
		if self.energy > 1:
			self.energy = 1


	def gather_res(self):
		# Arguable formula
	
		consumed = self.sim.env.consume_resource(self)
		self.inventory += consumed * res_to_invent * (colab_bonus * self.colab + 1)

		# Note Collaboration makes resource gathering more efficient
		

	# ------- Social decisions and actions ------- 

	
	def add_to_memory(self, agent_id):
		# Function that creates memory instance of internal state at the time of interaction with a certain agent
		# It is used to judge wether interaction with agent has proven useful afterwards
		self.memory.append((agent_id, self.maslow))


	def learn_about(self, from_agent_id, about_agent_id, score):

		#print('Ag', self.ident, ' learned about ag', about_agent_id, ' from ag', from_agent_id, ' (', score, ')')
		if from_agent_id in self.acquaint.keys():
			if about_agent_id in self.acquaint.keys():
				self.acquaint[about_agent_id] += round(score * self.acquaint[from_agent_id],2)
			else:
				self.acquaint[about_agent_id] = round(score * self.acquaint[from_agent_id],2)


	def train(self ,agent, iterations = 10):

		self.brain.mirror_training(agent.brain, batches = 10, epochs = 5)


	# ------- Cyclic parameter updates ------- 
	
	def refresh_memory(self):
		# Necessary for maintaining a constant memory queue
		if len(self.memory) > self.max_memory:
			self.memory.pop(0)


	def update_maslow(self, base = 2):
		#Computing the Score of necesities
		
		#Arguable formula
		self.maslow = base ** 2 * self.energy + base * self.social + self.actualization
		self.maslow = scale_val(self.maslow, self.sim.env.max_maslow, 1)


	def update_social(self):
		# Part of Maslow score that measures closeness to friends and distance to foes
		# Only applies to agents known by self
		new_social = 0

		for agent_id in self.acquaint.keys():
			
			try:
				agent = self.sim.all_agents[agent_id]

				# Argueable formula
				# Social closeness - the closer to friendly agents the better
				dist = distance(self.position, agent.position)
				if dist < 1:
					dist = 1

				if dist < self.sight:
					new_social = (1 / dist) * self.acquaint[agent.ident]
			except:
				pass

		self.social = round((self.social + new_social ) / 2, 2)
		
		# Value validation
		if self.social < 0:
			self.social = 0

		# Rescale value to 0-1
		self.social = scale_val(self.social, self.sim.env.max_social, 1)


	def update_reputation(self):
		# Metric of the Social Scores given to the agent by all the other agents
		rep = 0
		no = 0
	
		for agent_id, agent in self.sim.all_agents.items():
			if self.ident in agent.acquaint.keys():
				rep += agent.acquaint[self.ident]
				no += 1 

		if no:
			rep /= no

		self.reputation = scale_val(round(rep,2), self.sim.env.max_reputation, 1)


	def update_actualization(self):

	
		# Arguable formula

		if self.sim.env.monolith.seen == False:
			
			self.actualization = (self.actualization + self.reputation) / 2

			for ch_id in self.offsprings:

				if ch_id not in self.sim.all_agents.keys():
					# Child is dead
					self.actualization = 0
					self.offsrpings.remove(ch_id)
				else:
					# Parent wants to maximize offspring Maslow Score
					self.actualization = (self.actualization + self.sim.all_agents[ch_id].maslow) / 2

			# If the monolith is presemt, actualization is getting closer to it


		
		self.actualization = scale_val(self.actualization, self.sim.env.max_actualization, 1)

	
	def update_acquaint(self, base = 2):

		for mem in self.memory:

			agent_id, init_maslow = mem

			maxim = self.maslow * base ** self.memory.index(mem)
			interact_score = (self.maslow - init_maslow) * base ** self.memory.index(mem) / maxim 

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
		try:

			agent_id = roulette_selection(self.acquaint)
			
			if agent_id:
				score = self.acquaint[agent_id] / 10
				
				for other_agent_id in self.acquaint.keys():

					if other_agent_id != agent_id:

						other_agent = self.sim.all_agents[other_agent_id]
						other_agent.learn_about(self.ident, agent_id, score)
		except:
			pass
			

	def update_family(self, agent, relation):

		self.family[agent.ident] = relation
		for oth_agent_id, oth_relation in agent.get_family().items():
			self.add_family_member(oth_agent_id, relation * oth_relation)


	def add_family_member(self, agent_id, relation):
	
		if agent_id not in self.family.keys():
			self.family[agent_id] = relation
		else:
			if relation > self.family[agent_id]:
				self.family[agent_id] = relation


	def get_family(self):
		#
		return self.family


	def update_color(self):
		# Updates color of agent as a means of social ignalling:
		# Cooperating agents will converge to simmilar colors
		
		for agent_id, score in self.acquaint.items():

			try:

				agent = self.sim.all_agents[agent_id]
				oth_color = agent.color

				dist = distance(self.position, agent.position) + 0.1
				if dist:
					if dist < self.sight and abs(score) > 0.5:
						
						# Argueable formula
						color_delta = [(oth_c - my_c) / 255 for oth_c, my_c in zip(oth_color, self.color)]
						
						# Arguable formula
						#self.color = [my_c + c_delta * score / oth_c for my_c, c_delta, oth_c in zip(self.color, color_delta, agent.color)]
						self.color = [my_c + c_delta * score / dist for my_c, c_delta, oth_c in zip(self.color, color_delta, oth_color)]

						self.color = [0 if c < 0 else c for c in self.color]
						self.color = [1 if c > 1 else c for c in self.color]
			except:
				pass


# ----------- Interactions Class ----------- 

class Sap_Interactions:

	# ------- Pairing of agents ------- 

	def __init__(self, sim):
		# 

		self.sim = sim


	def simulate_interactions(self):

		for ag_id, agent in self.sim.all_agents.items():
			agent.colab = 0

		interacts = []
		agents_ids = list(self.sim.all_agents.keys())
		n = len(agents_ids)

		matches = np.full((n,n), np.inf)

		for i in range(n - 1):
			ag = self.sim.all_agents[agents_ids[i]]
			
			for j in range(i + 1, n):
				oth_ag = self.sim.all_agents[agents_ids[j]]
				d = distance(ag.position, oth_ag.position)
				
				if d < ag_max_interact_dist:
					matches[i,j] = d

		# Adding randomness to pairing
		#matches *= np.random.rand(n,n)

		pairs = []
		i = 0 

		for i in range(int(n/2)):

			if np.argmin(matches, axis=None) != np.inf:
				i, j = np.unravel_index(np.argmin(matches, axis=None), matches.shape)

				matches[i,:] = np.full(n, np.inf)
				matches[:,i] = np.full(n, np.inf).T
				matches[j,:] = np.full(n, np.inf)
				matches[:,j] = np.full(n, np.inf).T
		
				if agents_ids[i] != agents_ids[j]:
					pairs.append((agents_ids[i], agents_ids[j]))
					self.ag_interact(agents_ids[i], agents_ids[j])

		return pairs


	def ag_interact(self, agent1_id, agent2_id):

		agent1 = self.sim.all_agents[agent1_id]
		agent2 = self.sim.all_agents[agent2_id]
		
		f_1, c_1 = agent1.decide(agent2)
		f_2, c_2 = agent2.decide(agent1)

		if f_1 and c_1 and f_2 and c_2:
			# Both want to interact

			if f_1 > c_1 or f_2 > c_2:
				self.fight(agent1, f_1, agent2, f_2)
			else:
				if c_1 * c_2 > ag_mating_score and self.mating_compatibility(agent1, agent2):
					self.mate(agent1, agent2)
				else:
					self.collaborate(agent1, agent2)
			
			agent1.add_to_memory(agent2_id)
			agent2.add_to_memory(agent1_id)
		
		else:
			pass


	# ------- Possible interactions ------- 
	
	def fight(self, agent1, f1, agent2, f2):

		options = {}
		options[agent1.ident] = f1 * agent1.energy
		options[agent2.ident] = f2 * agent2.energy

		winner = roulette_selection(options)

		# Both agents lose energy 
		agent1.energy /= (1 + f1)
		agent2.energy /= (1 + f2)
		
		# Winner takes part of the loser's inventory
		def fight_result(winner, loser):
			winner.inventory += loser.inventory * invent_fract
			loser.inventory *= (1 - invent_fract)
		
		if agent1 is winner:
			fight_result(agent1, agent2)
		else:
			fight_result(agent2, agent1)

		info = 'Ag{} ({}) fought Ag{} ({})'.format(agent1.ident, np.round(f1,1), agent2.ident, np.round(f2,1))
		l.log(info)

		agent1.log.log('Fought with AG{}'.format(agent2.ident))
		agent2.log.log('Fought with AG{}'.format(agent1.ident))
	
	
	def collaborate(self, agent1, agent2):

		#agent1.colab = 1
		#agent2.colab = 2

		if agent1.maslow > agent2.maslow:
			agent1.train(agent2)
			agent1.inventory += agent2.inventory * invent_fract
			agent2.inventory *= (1 - invent_fract)

		else:
			agent2.train(agent1)
			agent2.inventory += agent1.inventory * invent_fract
			agent1.inventory *= (1 - invent_fract)

		info = 'Ag{} helped Ag{}'.format(agent1.ident, agent2.ident)
		l.log(info)

		agent1.log.log('Collaborated with AG{}'.format(agent2.ident))
		agent2.log.log('Collaborated with AG{}'.format(agent1.ident))
	

	def mate(self, agent1, agent2):
		

		pos = [int((p1 + p2) / 2) for p1, p2 in zip(agent1.position, agent2.position)]

		newborn = Sap(self.sim, pos, agent1, agent2)
		self.sim.add_agent(newborn)

		info = 'NEWBORN - Ag{} (Ag{} and Ag{}) with DNA: {}'.format(newborn.ident, agent1.ident, agent2.ident, newborn.dna)
		l.log(info)

		newborn.acquaint[agent1.ident] = 1
		newborn.acquaint[agent2.ident] = 1

		agent1.acquaint[newborn.ident] = 1
		agent2.acquaint[newborn.ident] = 1

		agent1.train(newborn)
		agent2.train(newborn)

		agent1.log.log('Mated with AG{}'.format(agent2.ident))
		agent2.log.log('Mated with AG{}'.format(agent1.ident))

		# Update family
		# Parent-child relation is 0.9 (1 would not vanish through family tree)
		newborn.update_family(agent1, 0.9)
		newborn.update_family(agent2, 0.9)

		agent1.add_family_member(newborn.ident, 0.9)
		agent2.add_family_member(newborn.ident, 0.9)

		# Note: Parents actualization scores include Maslow score of child

	# ------- Utils ------- 

	def mating_compatibility(self, agent1, agent2):

		rel1 = 0
		if agent1.ident in agent2.family.keys():
			rel1 = agent2.family[agent1.ident]

		rel2 = 0
		if agent2.ident in agent1.family.keys():
			rel2 = agent1.family[agent2.ident]

		related = max(rel1, rel2)

		diff = self.compute_agents_difference(agent1, agent2)

		if related < ag_blood_relation_thresh and diff < ag_compatibility_thresh:
			print('Ag{} and Ag{} compatible for mating.'.format(agent1.ident, agent2.ident))
			return True

		else:
			print('Ag{} and Ag{} NOT compatible for mating.'.format(agent1.ident, agent2.ident))
			return False


	def compute_agents_difference(self, agent1, agent2, batches = 10):
		
		# Computes decisional distance of agents
		
		diff = 0 

		for i in range(batches):

			diff += self.agents_decision_distance(agent1, agent2)

		diff /= batches

		print('Simmilarity result for Ag{} and Ag{}: {}'.format(agent1.ident, agent2.ident, np.round(diff,2)))

		return diff


	def agents_decision_distance(self, agent1, agent2):
		# Computes difference between 2 agents given a random input instance

		inp = np.random.rand(agent1.brain.no_inputs).reshape(1, agent1.brain.no_inputs)

		resp1 = agent1.brain.one_batch_action(inp)
		resp2 = agent2.brain.one_batch_action(inp)

		diff = np.sum(abs(resp1 - resp2))

		return diff


	def instance_decision_distance(self, agent, inp, out):

		resp = agent.brain.one_batch_action(inp)

		return abs(resp - out)





		





