import math
import numpy as np
import os


class Utils:
	def __init__(self):
		pass

	def scale_val(self, val, old_max, new_max, old_min = 0, new_min = 0):
		return ((val - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min

	def scale_arr(self, arr,new_max, new_min = 0):
		old_max = arr.max()
		old_min = arr.min()
		return ((arr - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min

	def normalize(self, arr, new_min = 0, new_max = 1):
		if arr.min() < 0:
			arr -= arr.min()
		arr /= arr.max()

		arr *= (new_max - new_min)
		arr += new_min
		
		return arr

	def distance(self, pos1,pos2):
		x1, y1 = pos1
		x2, y2 = pos2
		dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
		return dist   

	def roulette_selection(self, choices):
		# Choices is a dictionary of {Choice, Choice_Weight} pairs

		maximum = sum(abs(value) for value in choices.values())
		
		pick = np.random.uniform(0,maximum)
		curr = 0
		for key, value in choices.items():
			curr += abs(value)
			if curr > pick:
				return key

	def gradient(self, arr, x, y):

		# OX Gradient
		d_x = 0
		try: 
			d_x += arr[x + 1][y]
		except:
			pass
		try:
			d_x -= arr[x - 1, y]
		except:
			pass

		# OY Gradient
		d_y = 0
		try: 
			d_y += arr[x, y + 1]
		except:
			pass
		try:
			d_y -= arr[x, y - 1]
		except:
			pass

		if not d_x:
			d_x = 0
		if not d_y:
			d_y = 0

		return d_x, d_y
	

class Logger:
	def __init__(self, log_file):
		self.log_file = log_file

		if not os.path.isfile(self.log_file):
			with open(self.log_file, 'w') as file:
				file.write('Simmulation Log: \n')

	def log(self, text):
		with open(self.log_file, 'a') as file:
			file.write(text + '\n')

	def reset(self):
		with open(self.log_file, 'w') as file:
				file.write('Simmulation Log: \n')

class Genetics:
	def __init__(self):
		pass

	def get_color(self, parent1, parent2):

		if parent1 != None and parent2 != None:
			return [(c1 + c2 )/ 2 for c1, c2 in zip(parent1.color, parent2.color)]

		else:
			return list(np.random.uniform(size = 3))

	def get_dna(self, parent1, parent2, no_inputs, no_outputs):
		dna = []
		if parent1 != None and parent2 != None:

			for i in range(min(len(parent1.dna), len(parent2.dna))):
				mut = np.random.randint(-1,2)
				if np.random.uniform() > 0.5:
					if int(parent1.dna[i] + mut) < 2:
						dna.append(2)
					else:
						dna.append(int(parent1.dna[i] + mut))
				else:
					if int(parent2.dna[i] + mut) < 2:
						dna.append(2)
					else:
						dna.append(int(parent2.dna[i] + mut))

			if max(len(parent1.dna), len(parent2.dna)) != min(len(parent1.dna), len(parent2.dna)):
				dna.append(np.random.randint(0.5 * no_outputs, 1.5 * no_outputs + 1))

			if np.random.uniform() > 0.95:
				dna.append(np.random.randint(0.5 * no_outputs, 1.5 * no_outputs + 1))

		else:
			dna.append(np.random.randint(0.5 * no_inputs, 1.5 * no_inputs + 1))
			dna.append(np.random.randint(0.5 * no_outputs, 1.5 * no_outputs + 1))

		return dna


