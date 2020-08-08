#For drawing Bezier Curve

import numpy as np
import random
from scipy.interpolate import barycentric_interpolate
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from Agent import Sap


# ----------- Environment Parameters ----------- 

env_size = 600

env_max_danger = 10
env_max_resource = 100
env_max_resource_gen_rate = 10

resource_consumption_rate = 0.01

patch_size = 50
patch_space = 10


# ----------- Costum Image handling functions ----------- 

def show_image(image):
	#PIL needs image to be flipped
	ImageOps.flip(image).show()

def save_image(image, file):
	#PIL needs image to be flipped
	ImageOps.flip(image).save(file)


# ----------- Sap Agent Image ----------- 

def import_sap_symbol(file):
	symbol = Image.open('Images/Sap.png').convert('RGBA')
	return symbol

def colour_sap(symbol,colour = None):

	#Generate random colour if none provided
	if colour == None:
		colour = list(np.random.choice(range(256), size=3))

	#Unpacking image channels
	data = np.array(symbol)   			
	red, green, blue, alpha = data.T 	
	white_areas = (red == 255) & (blue == 255) & (green == 255)
	
	#Recolour white area
	data[..., :-1][white_areas.T] = colour

	#Generate recoloured Sap symbol
	symbol = Image.fromarray(data)
	symbol = ImageOps.flip(symbol.resize((patch_size, patch_size), Image.ANTIALIAS))

	return symbol

def show_sap(image, symbol, position):
	(x,y) = position 

	#Adding variation to position
	var = int(np.random.uniform(-1, 1) * patch_size) / 2 
	position = (int(x + var), y)
	
	img.paste(tok, position, mask = tok)


# ----------- Patch Class ----------- 

class Patch:
	def __init__(self, position, danger = 0, resource_gen_rate = 0, population = []):
		self.position = position

		if resource_gen_rate != 0 :
			self.resource_gen_rate = resource_gen_rate
		else:
			self.resource_gen_rate = np.random.randint(env_max_resource_gen_rate + 1)

		self.resource = env_max_resource * np.random.rand()

		if danger != 0:
			self.danger = danger
		else:
			self.danger = np.random.randint(env_max_danger + 1)

		self.population = population

	def grow_resource(self):
		self.resource = self.resource * (1 + resource_gen_rate) * ( env_max_resource - self.resource)

	def get_resource(self, efficiency = 1):
		luck = np.random.uniform(0.5,1.5)
		quant = self.resource * luck * efficiency * resource_consumption_rate
		self.resource -= quant
		return quant

	def draw(self, drawing):
		drawing.rectangle((self.position, 0, self.position + patch_size,  self.resource), fill=(0, 128, 0), outline=(0, 100, 0))

	def 
		

class Environment:
	def __init__(self, size = 100):

		self.sap = import_sap_symbol('Images/Sap.png')
		unit = patch_size + patch_space
		no_patches = int(size / unit)
		self.patches = []

		for i in range(no_patches):
			self.patches.append(Patch(position = i * unit))

	def draw(self,image):
		
		drw = ImageDraw.Draw(image)
		
		for patch in self.patches:
			patch.draw(drw)
			for i in range(np.random.randint(5) + 1):
				draw_sap(image, position = (patch.position, int(patch.resource)))

		show_image(image)
		save_image(image,'Env.jpg')


env = Environment(env_size)

image = Image.new('RGB', (env_size, 2 * env_max_resource))

env.draw(image)

