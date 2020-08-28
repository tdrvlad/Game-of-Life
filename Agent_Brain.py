# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from collections import deque

import os
import yaml

os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #This makes it run on CPU (no error)

#tf.config.run_functions_eagerly(True)

parameter_file = 'Parameters.yaml'

param = yaml.load(open(parameter_file), Loader = yaml.FullLoader)
training_batch = param['training_batch']


class Brain:

	def __init__(self, no_inputs, no_outputs, architecture):
		
		# Architecture (n,m,n) means a neural network of 3 fully connected layers with n , m and n neurons respectevley.

		self.no_inputs = no_inputs
		self.no_outputs = no_outputs
		self.no_layers = len(architecture)
		self.neurons_per_layer = architecture

		self.gamma = 0.95
		self.epsilon = 5.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.1
		self.tau = .05

		self.model = self.create_model()
		self.target_model = self.create_model()

		self.memory  = deque(maxlen=2000)
		

	def create_model(self):

		model = Sequential()
		
		model.add(Input(self.no_inputs, name = 'Input'))
		
		for i in range(self.no_layers):
			model.add(Dense(self.neurons_per_layer[i], activation = 'sigmoid', name = 'Hidden' + format(i)))
		model.add(Dense(self.no_outputs, activation = 'sigmoid', name = 'Output'))
		
		model.compile(
			optimizer = Adam(lr = self.learning_rate), 
			loss = 'categorical_crossentropy', 
			metrics = ['accuracy'],
			)

		return model

	def replay(self):
		if len(self.memory) < training_batch: 
			return

		samples = random.sample(self.memory, training_batch)

		for sample in samples:
			
			state, action, reward, new_state = sample

			target = self.target_model.predict(state)
			
			#Rounding actions to either -1, 0 or 1
			action = np.rint(action).astype(int)
			
			target[0][action] = reward
			
			Q_future = max(self.target_model.predict(new_state)[0])
			target[0][action] = reward + Q_future * self.gamma
			
			self.model.fit(state, target, epochs=1, verbose=0)
			
			
	def target_train(self):
		weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
		self.target_model.set_weights(target_weights)

	def remember(self, state, action, reward, new_state):
		# 
		self.memory.append([state, action, reward, new_state])

	def decide(self, state):
		self.epsilon *= self.epsilon_decay
		self.epsilon = max(self.epsilon_min, self.epsilon)
		if np.random.random() < self.epsilon:
			return np.random.uniform(0, 1, self.no_outputs)
		return self.model.predict(state)[0]  

	def save_model(self, file):
		self.model.save(file)

	def summary(self):
		self.model.summary()


if __name__ == '__main__':

	brain = Brain(2,2,(3,3))
	brain.summary()
