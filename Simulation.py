import numpy as np
#from Environment import Environment
from Agent import Sap
from Environment import Environment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numpngw import write_apng

import gif

import math

def distance(pos1,pos2):
	x1, y1 = pos1
	x2, y2 = pos2
	dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
	return dist  

image_file = 'Env_Snapshots/Tick'
env_agents = {}

def add_agent(agent, all_agents):
	
	if bool(all_agents) :
		if agent.ident == None:
			agent.ident = max(all_agents, key=int) + 1
		
	else:	
		agent.ident = 0

	all_agents[agent.ident] = agent

dim = 400		
env = Environment((dim,dim))

no_agents = 35
for i in range(no_agents):
	x = int(np.random.randint(0,dim))
	y = int(np.random.randint(0,dim))
	add_agent(Sap((x,y)), env_agents)



no_ticks = 300

for t in range(no_ticks):
	
	print('Simulation day ',t)
	file = image_file + str(t) + '.png'
	plot = env.draw_environment(env_agents, file)

	for agent_id, agent in env_agents.items():
		for oth_agent_id, oth_agent in env_agents.items():
			if distance(agent.position, oth_agent.position) < math.sqrt(dim):
				agent.update_acquaint(oth_agent_id, np.random.uniform(-1,1))
				oth_agent.update_acquaint(agent_id, np.random.uniform(-1,1))

	for agent_id, agent in env_agents.items():
		agent.life_tick(env_agents,env)

	env.regen_resource()

	


plt.clf()

fig = plt.figure()

images = []

print('Recomposing animation')
for t in range(no_ticks):
	file = image_file + str(t) + '.png'
	im = plt.imread(file)
	images.append([plt.imshow(im)])

ani = animation.ArtistAnimation(fig, images, interval=200, repeat_delay=0)
plt.axis('off')

writergif = animation.PillowWriter(fps=12) 
ani.save('Evolution.gif', writer=writergif)
#plt.show()
#plt.savefig('Evolution.png')
