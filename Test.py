import numpy as np


n = 5

matches = np.full((n,n), np.inf)

for i in range(n-1):
			
	for j in range(i + 1, n):
		
		d = np.random.uniform(5,10)
				
		matches[i,j] = d


print(matches)
print(np.unravel_index(np.argmin(matches, axis=None), matches.shape))		
'''
		# Adding randomness to pairing
pairs = []
i = 0 
while matches.shape[0] > 2:

	print('Matix:')
	print(matches)

	j = np.argmin(matches, axis = 1)[0]
	pairs.append((i,j + i))

	i += 1

	matches = matches[1:,1:]


print(pairs)'''