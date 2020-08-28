import math
import numpy as np

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

		return d_x, d_y
	

