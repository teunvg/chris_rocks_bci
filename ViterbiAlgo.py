import numpy as np
from random import randint, uniform

# generate an example dataset
from docutils.nodes import emphasis

''' 3 states: healthy (0), sick (1), in recovery (2). But not directly observable. Instead we have a set of observations
from which we "guess" the states. For now we just randomly deviate from the right state with a probability.
'''
def generateData(nStates, nReps, emissionMatrix):
    minRepetitions = nReps - 6
    maxRepetitions = nReps + 6
    observations = []
    for state in range(0, nStates):
        for repetition in range(0, randint(minRepetitions, maxRepetitions)):
            chance = uniform(0, 1)
            nextState = (state + 1) % nStates
            prevState = (state + nStates - 1) % nStates
            if chance <= emissionMatrix[state][state]:
                observations.append(state)
            elif chance <= emissionMatrix[state][nextState]:
                observations.append(nextState)
            else:
                observations.append(prevState)
    return observations

nStates = 3
nReps = 19.  # average number of time moments a state is repeated
pRight = 0.8  # probability of estimating the state wrongly
pNextState = 0.13  # probability of miss-classifying as the next state
O = np.array([0, 1, 2])  # observation space (in our case obs = state as estimated by the classifier)
S = np.array([0, 1, 2])  # state space (0=healthy, 1=sick, 2=in recovery) we assume it can only go 0-->1-->2
Pi = np.array([0.34, 0.33, 0.33])  # (greek PI) initial probabilities; assumes all events last the same
transitionMatrix = np.array([[nReps/(nReps+1), 1/(nReps+1), 0], \
                             [0, nReps/(nReps+1), 1/(nReps+1)], \
                             [1/(nReps+1), 0, nReps/(nReps+1)]])  # probabilities of transitioning between states
emissionMatrix = np.array([[pRight, pNextState, 1-pRight-pNextState], \
                           [1-pRight-pNextState, pRight, pNextState], \
                           [pNextState, 1-pRight-pNextState, pRight]])  # probabilities of observing a state (given the state)
obvs = np.array(generateData(nStates, nReps, emissionMatrix))  # a list of observations (states as estimated by the classifier)

Tcur = np.zeros([nStates, obvs.size])
Tmax = np.zeros([nStates, obvs.size])
for i in S:
    Tcur[i][0] = Pi[i] * emissionMatrix[i][0]
    Tmax[i][0] = 0

for i in range(1, obvs.size):
    for j in S:
        Tcur[j][i] = emissionMatrix[j][obvs[i]]
















