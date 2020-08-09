import numpy as np
import random
from scipy.interpolate import barycentric_interpolate
import matplotlib.pyplot as plt


class Sap:
	def __init__(self,position,color = None):
		self.position = position

		self.needs 
		self.comfort 
		self.actualization

		self.inventory = 0
		self.age = 0
		self.offspring = []



		if color != None:
			self.color = color
		else:
			self.color = list(np.random.choice(range(256), size=3) / 255)
