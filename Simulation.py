import numpy as np
#from Environment import Environment
from Agent import Sap
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
	def __init__(self, dimension = 200 ,no_agents = 10 ,time_units = 40):
		self.dimension = dimension
		self.no_agents = no_agents
		self.time_units = time_units

		self.all_agents = {}
		self.agents_to_remove = []



	def add_agent(self, agent):
		# Function for adding a new agent to the list of all previous agents
		if bool(self.all_agents) :
			if agent.ident == None:
				agent.ident = max(self.all_agents, key=int) + 1
			
		else:	
			agent.ident = 0

		self.all_agents[agent.ident] = agent

	def init_sim(self):

		self.env = Environment((self.dimension,self.dimension))

		for i in range(self.no_agents):
			x = int(np.random.randint(0,self.dimension))
			y = int(np.random.randint(0,self.dimension))
			self.add_agent(Sap((x,y)))

	def run_sim_unit(self, t, save_path):

		file = save_path + str(t) + '.png'
		self.env.draw_environment(self, file)

		for agent_id, agent in self.all_agents.items():
			for oth_agent_id, oth_agent in self.all_agents.items():
				if distance(agent.position, oth_agent.position) < math.sqrt(self.dimension):
					agent.update_acquaint(oth_agent_id, np.random.uniform(-1,1))
					oth_agent.update_acquaint(agent_id, np.random.uniform(-1,1))

		for agent_id, agent in self.all_agents.items():
			alive = agent.life_tick(self)
				
			if not alive:
				self.agents_to_remove.append(agent_id)
				print('Sap ', agent_id, ' died.')

		for agent_id in self.agents_to_remove:
			del self.all_agents[agent_id]

		self.agents_to_remove = []

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

	def run_all(self, save_path):

		self.init_sim()

		for t in range(self.time_units):
			self.run_sim_unit(t, save_path)

		self.animate_evolution(save_path)


if __name__ == '__main__':

	sim = Simulation(time_units = 50, no_agents = 25)
	sim.run_all(image_file)