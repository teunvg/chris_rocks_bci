import numpy as np
import seaborn as sns
from buffer_bci import preproc
import pickle

events = data = []

with open('chris_data_2', 'r') as file:
    data, events = pickle.load(file).values()

types, startData, endData = [], [], []
for event, datum in zip(events, data):
    if event.type == 'stimulus.tgtFlash':
        startData.append(datum)
        types.append(event.value)
    elif event.type == 'stimulus.tgtHide' and len(startData) == len(endData) - 1:
        endData.append(datum)

print(len(types))