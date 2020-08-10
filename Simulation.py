import numpy as np
#from Environment import Environment
from Agent import Sap
from Environment import Environment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numpngw import write_apng

import gif

image_file = 'Env_Snapshots/Tick'
env_agents = {}

def add_agent(agent, all_agents):
	
	if bool(all_agents) :
		if agent.ident == None:
			agent.ident = max(all_agents, key=int) + 1
		
	else:	
		agent.ident = 0

	all_agents[agent.ident] = agent

dim = 200		
env = Environment((dim,dim))

no_agents = 2
for i in range(no_agents):
	x = int(np.random.randint(0,dim))
	y = int(np.random.randint(0,dim))
	add_agent(Sap((x,y)), env_agents)



no_ticks = 10

for t in range(no_ticks):
	x = np.random.randint(200)
	y = np.random.randint(200)

	add_agent(Sap((x,y)), env_agents)
	env.consume_resource((x,y))

	file = image_file + str(t) + '.png'
	plot = env.draw_environment(env_agents, file)
fig = plt.figure()

images = []
for t in range(no_ticks):
	file = image_file + str(t) + '.png'
	im = plt.imread(file)
	images.append([plt.imshow(im)])

ani = animation.ArtistAnimation(fig, images, interval=500, repeat_delay=1000)
plt.show()
