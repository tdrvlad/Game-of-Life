
import tensorflow as tf

import sys

sys.stdout.flush()

import numpy as np
#from Environment import Environment
from Agent import Sap, Sap_Interactions

from Environment import Environment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Utils import Utils
utils = Utils()
normalize = utils.normalize
distance = utils.distance
scale_val = utils.scale_val

from numpngw import write_apng

import gif

import math


image_file = 'Env_Snapshots/Tick'

class Simulation:
	def __init__(self, dimension = 200 ,no_agents = 20 ,time_units = 40):

		#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1 / (no_agents + 1))
		#sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

		self.dimension = dimension
		self.no_agents = no_agents
		self.time_units = time_units

		self.all_agents = {}
		self.agents_to_remove = []

		interact = Sap_Interactions(self)


	def add_agent(self, agent):
		# Function for adding a new agent to the list of all previous agents
		if bool(self.all_agents) :
			if agent.ident == None:
				agent.ident = max(self.all_agents, key=int) + 1
			
		else:	
			agent.ident = 0

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

		self.env = Environment((self.dimension,self.dimension))

		for i in range(self.no_agents):
			x = int(np.random.uniform(2 * self.dimension / 10, 9 * self.dimension / 10))
			y = int(np.random.uniform(2 * self.dimension / 10, 9 * self.dimension / 10))
			self.add_agent( Sap(self, (x,y) ) )

		for agent_id, agent in self.all_agents.items():
			oth_agent_id = np.random.choice(list(self.all_agents.keys()))
			agent.acquaint[oth_agent_id] = np.random.uniform(-1,1)

			oth_agent_id = np.random.choice(list(self.all_agents.keys()))
			agent.acquaint[oth_agent_id] = np.random.uniform(-1,1)


	def run_sim_unit(self, t, save_path, visualize):

		print('Simtime: {}'.format(t), flush = True)
		if visualize:
			file = save_path + str(t) + '.png'
			self.env.draw_environment(self, file)
		else:
			self.env.draw_environment(self)

		for agent_id, agent in self.all_agents.items():
			alive = agent.life_tick()
				
			if alive == 0:
				self.agents_to_remove.append(agent_id)

		self.env.regen_resource()


	def animate_evolution(self, save_path):

		plt.clf()

		fig = plt.figure()

		images = []

		print('Recomposing animation')
		for t in range(self.time_units):
			file = image_file + str(t) + '.png'
			im = plt.imread(file)
			images.append([plt.imshow(im)])

		ani = animation.ArtistAnimation(fig, images, interval=100, repeat_delay=0)
		plt.axis('off')

		writergif = animation.PillowWriter(fps=5) 
		ani.save('Evolution.gif', writer=writergif)
		
		#plt.show()


	def run_all(self, save_path, visualize = False):

		self.init_sim()

		for t in range(self.time_units):
			self.run_sim_unit(t, save_path, visualize = visualize)
			#self.delete_agents()
		if visualize:
			self.animate_evolution(save_path)


if __name__ == '__main__':

	import time
	start_time = time.time()

	sim = Simulation(time_units = 25, no_agents = 3)
	sim.run_all(image_file, visualize = True)

	print('Run time: %s seconds' % (time.time() - start_time))
