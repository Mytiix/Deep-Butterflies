import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
	with open('sp.pkl', 'rb') as file:
		sp = pickle.load(file)
		
		species = [k for k, v in sp.items()]
		numbers = [max(len(v[0]),len(v[1])) for k, v in sp.items()]

		plt.figure(figsize=(12,10))
		plt.bar(species, numbers, color='orange')
		plt.ylabel('Number of samples')
		plt.xlabel('Subspecies')
		plt.xticks(rotation=45)
		plt.savefig('Subspecies_distribution.png')