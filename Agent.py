import numpy as np
import random
from scipy.interpolate import barycentric_interpolate
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps

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






class Sap:
	def __init__(self):
