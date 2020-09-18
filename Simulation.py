import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import time
import os
import glob
import yaml

from numpngw import write_apng
import gif

from Agent import Sap, Sap_Interactions
from Environment import Environment
from Utils import Utils, Logger

utils = Utils()
normalize = utils.normalize
distance = utils.distance
scale_val = utils.scale_val

#--------------------------------

parameter_file = 'Parameters.yaml'
agent_log_dir = 'Agents_Logs'
snap_dir = 'Snapshots'
snap_file = snap_dir + '/Tick'
gif_file = 'Evolution.gif'

param = yaml.load(open(parameter_file), Loader = yaml.FullLoader)
monolith_spawn_chance = param['monolith_spawn_chance']

#--------------------------------

class Simulation:
	def __init__(self, parameter_file):

		param = yaml.load(open(parameter_file), Loader = yaml.FullLoader)
		self.time_units = param['time_units']
		self.no_agents = param['no_agents']

		try:
			time_units = sys.argv[1]
			no_agents = sys.argv[2]
		except:
			pass

		self.parameter_file = parameter_file
		self.all_agents = {}
		self.agents_to_remove = []

		self.sim_interactions = Sap_Interactions(self).simulate_interactions

		self.time = 0


	def add_agent(self, agent):
		# Function for adding a new agent to the list of all previous agents
		if bool(self.all_agents) :
			if agent.ident == None:
				agent.ident = max(self.all_agents, key=int) + 1
			
		else:	
			agent.ident = 0

		agent.log = Logger(agent_log_dir + '/Agent' + str(agent.ident) + '.txt')

		print('New agent: Ag{}'.format(agent.ident))
		self.all_agents[agent.ident] = agent


	def delete_agents(self):

		for agent_id, agent in self.all_agents.items():
			for ag_id in self.agents_to_remove:
				agent.acquaint.pop(ag_id, None)

		for agent_id in self.agents_to_remove:
			print('Sap ', agent_id, ' died.')
			self.all_agents.pop(agent_id, None)

		self.agents_to_remove = []


	def init_sim(self):

		self.env = Environment(self.parameter_file)
		x_max = self.env.dimension_x
		y_max = self.env.dimension_y

		for i in range(self.no_agents):
			x = int(np.random.uniform(2 * x_max / 10, 9 * x_max / 10))
			y = int(np.random.uniform(2 * y_max / 10, 9 * y_max / 10))

			self.add_agent( Sap(self, (x, y) ) )

		for agent_id, agent in self.all_agents.items():
			oth_agent_id = np.random.choice(list(self.all_agents.keys()))
			agent.acquaint[oth_agent_id] = np.random.uniform(-1,1)

			oth_agent_id = np.random.choice(list(self.all_agents.keys()))
			agent.acquaint[oth_agent_id] = np.random.uniform(-1,1)


	def run_sim_unit(self, visualize):

		self.time += 1

		print('Sim Time Unit {}'.format(self.time), flush = True)

		# Monolith - external motivator 

		#if self.env.monolith.seen == False and np.random.uniform() < monolith_spawn_chance:
		if self.env.monolith.seen == False and self.time > 50:
			pos = np.random.randint(self.env.dimension_x), np.random.randint(self.env.dimension_y) 
			self.env.monolith.spawn(pos)

		self.env.monolith.radiate(self)

		# Environment Parameters

		self.env.update_danger(self)
		self.env.regen_resource()

		
		for agent_id, agent in self.all_agents.items():
			alive = agent.life_tick()
				
			if alive == 0:
				self.agents_to_remove.append(agent_id)

		self.delete_agents()

		interacts = self.sim_interactions()
		self.env.draw_environment(self, tick = self.time, interacts = interacts, image_file = snap_file + str(self.time).zfill(3) + '.png')
	

	def run_all(self, visualize = False):

		self.init_sim()

		start_time = time.time()

		for t in range(self.time_units):
			self.run_sim_unit(visualize = visualize)

					
			if len(list(self.all_agents.keys())) == 0:
				print('All Saps died')
				break

		print('Run time: %s seconds' % (time.time() - start_time), flush = True)

		if visualize:
			self.animate_evolution()


	def animate_evolution(self):

		try:
			no_snaps = len([name for name in os.listdir(snap_dir) if os.path.isfile(os.path.join(snap_dir, name))])

			plt.clf()

			fig = plt.figure(figsize = (30,30))

			images = []

			print('Recomposing animation ({} snapshots found)'.format(no_snaps), flush = True)
			
			files = glob.glob(snap_dir + '/*')
			for file in files:
				im = plt.imread(file)
				images.append([plt.imshow(im)])

			ani = animation.ArtistAnimation(fig, images, interval=100, repeat_delay=0)
			plt.axis('off')

			writergif = animation.PillowWriter(fps=5) 
			ani.save(gif_file, writer = writergif)
			
			#plt.show()
		except:
			pass


if __name__ == '__main__':

	log_file = 'Sim_Log.txt'
	l = Logger(log_file)
	l.reset()

	sim = Simulation(parameter_file)

	if not os.path.isfile(gif_file):
		sim.animate_evolution()

	else:
		try:

			files = glob.glob(snap_dir + '/*')
			for f in files:
				os.remove(f)

			files = glob.glob(agent_log_dir + '/*')
			for f in files:
				os.remove(f)
		except:
			pass

		if not os.path.isdir(snap_dir):
			os.mkdir(snap_dir)

		print('Running simmulation for {} Ticks with {} Agents.'.format(sim.time_units, sim.no_agents))

		sim.run_all(visualize = True)



