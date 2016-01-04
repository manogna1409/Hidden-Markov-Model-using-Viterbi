import numpy as np
import math
from sklearn.preprocessing import normalize

obs = []
states = []

f = open('/Users/manogna/Desktop/entrain.txt','rb')
line = f.readline()
while line!= '':
	parts = line.split('/')
	obs.append(parts[0].lower())
	states.append(parts[1])
	line = f.readline()

nStates = len(set(states))
nObs = len(obs)


ids = 0
seen = {'UNK':ids}
seen_rev = {ids:'UNK'}
ids+=1
for i in range(len(obs)):
	if not(obs[i] in seen):
		seen[obs[i]] = ids
		ids+=1
		obs[i] = 'UNK'


state_ids={}
state_ids_rev = {}
ids = 0
for i in range(len(states)):
	if states[i].strip() not in state_ids:
		state_ids[str(states[i].strip())] = ids
		state_ids_rev[ids] = str(states[i].strip())
		ids+=1


obs_prob = np.ones((nStates,len(seen)), dtype=float)
state_trans = np.ones((nStates,nStates), dtype=float)


for i in range(len(states)):
	x = state_ids[states[i].strip()]
	y = seen[obs[i]]
	obs_prob[x,y] += 1
	if i<len(states)-1:
		x = state_ids[states[i].strip()]
		y = state_ids[states[i+1].strip()]
		state_trans[x,y] += 1

obs_prob = normalize(obs_prob, norm='l1', axis=1)
state_trans = normalize(state_trans, norm='l1', axis=1)




test_obs = []
test_states = []

f = open('/Users/manogna/Desktop/entest.txt','rb')
line = f.readline()
while line!= '':
	parts = line.split('/')
	test_obs.append(parts[0].lower())
	test_states.append(parts[1].strip())
	line = f.readline()


for i in range(len(test_obs)):
	if not(test_obs[i] in seen):
		test_obs[i] = 'UNK'

# Use Viterbi algorithm for prediction.

# We will use a trick to find out most probable path. In order to ensure that multiplication
# of our probabilities doesn't become indistinguishably zero for computer to process, we will
# take log of probabilities. Then we can use log(p1 X p2) = log(p1) + log(p2).

max_path = []
k = seen[test_obs[0]]
state_max = [math.log((1.0/nStates)*obs_prob[i,k],2) for i in range(nStates)]

for i in range(1,len(test_obs)):
	mx = np.argmax(state_max)
	max_path.append(state_ids_rev[mx])
	curr_state_max = []

	# Convention:
	# x: Index for hidden states of current output.
	# y: Index for hidden states of previous output.

	for x in range(nStates):
		temp = []
		k = seen[test_obs[i]]
		for y in range(nStates):
			temp.append(state_max[y]+math.log(state_trans[y,x],2)+math.log(obs_prob[x,k],2))
		curr_state_max.append(max(temp))
	state_max = curr_state_max

mx = np.argmax(state_max)
max_path.append(state_ids_rev[mx])



# Calculate Error rate.
err = 0
for i in range(0,len(test_obs)):
	if test_states[i] != max_path[i]:
		err+=1
err = (err)*1.0/len(test_obs)


print ("Error rate = ", float('%0.4f'%(err*100)), "%")
