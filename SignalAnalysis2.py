import numpy as np
from scipy import signal
import seaborn as sns
from buffer_bci import preproc, linear
import pickle
import pywt as pywt
import sklearn as sk

events = data = []
with open('teun_im_live' + '.dat', 'r') as file:
    raw = pickle.load(file)
    hdr, data, events = raw['hdr'], raw['data'], raw['events']

observation = np.zeros((10,9))
transition = np.zeros((10,10))
typeClassifier, signalClassifiers = None, None
with open('l1_classifiers' + '.dat', 'r') as file:
    raw = pickle.load(file)
    observation, typeClassifier, signalClassifiers = raw['observation'], raw['type_classifier'], raw['signal_classifiers']

n_channels = 10
freq = 250.

data, events = data[576:], events[576:]
data = data[6:] # remove leading 1.5s of bogus data
events = events[:-6] # remove trailing six event bins
data = [datum[:,:n_channels] for i, datum in enumerate(data)]
data = preproc.detrend(data)
data = preproc.spatialfilter(data, type="car")
#data, events, badtrials = preproc.badtrialremoval(data, events)
data = [signal.spectrogram(datum, freq, nperseg=64, noverlap=54, axis=0) for datum in data]
bands = data[0][0]
deltas = [np.logical_and(bands > i, bands < j) for i, j in zip([0], [4])]
mus = [np.logical_and(bands > i, bands < j) for i, j in zip([7], [14])]
betas = [np.logical_and(bands > i, bands < j) for i, j in zip([14], [30])]
data = [np.array([datum[2][band,:,0:31].mean(axis=0).tolist() for band in deltas + mus + betas]) for datum in data]

# Normalise
flat_data = np.array([datum.flatten().tolist() for datum in data])
feature_means = flat_data.mean(axis=0)
demeaned_data = flat_data - feature_means
feature_stds = demeaned_data.std(axis=0)
norm_data = demeaned_data / feature_stds
data = [np.array(datum).reshape(3, -1, 31) for datum in norm_data.tolist()]

# Main data processing
type_data = [datum[:,:,0:31:5].reshape(1, -1) for datum in data]
signal_data = [
    [datum[0,:,0:6].reshape(1, -1) for datum in data],
    [datum[1:,:,0:31:5].reshape(1, -1) for datum in data],
    [datum[2,:,0:31:5].reshape(1, -1) for datum in data]
]

#predictions = [(np.array([signalClassifiers[signalClass].predict_proba(signal[i]) 
#                    for i, signalClass in enumerate(signal_data.keys())]).squeeze() * typeClassifier.predict_proba(type).T).flatten().tolist()
#                for type, signal in zip(type_data, zip(*signal_data.values()))]

type_names = ('LRP', 'ERD', 'ERS')
t_pred = [typeClassifier.predict(type)[0] for type in type_data]
predictions = [signalClassifiers[type_names[type]].predict(signal[type])[0] + 3 * type for type, signal in zip(t_pred, zip(*signal_data))]

#print(np.array(predictions).sum(axis=0))
print(np.histogram(predictions, bins=9))

# Turn events into indices
classes = ["RH", "LH", "F"]
types = ["stimulus.prepare", "stimulus.start", "stimulus.stop"]
events = [[event for event in event_list if event.type in types] for event_list in events]
event_types = np.array([(types.index(event_list[0].type) * 3 + classes.index(event_list[0].value) if len(event_list) > 0 else -1) for event_list in events])
event_types[np.logical_and(event_types >= 3, event_types < 6)] = 2
stop_events = np.where(event_types >= 6)[0]
event_types[stop_events[stop_events < event_types.size - 3] + 3] = -event_types[stop_events]-2
event_types[stop_events] = 2
event_types += 1

event_series = np.cumsum(event_types)
event_series[event_series == 0] = 10
event_series -= 1
print(np.histogram(event_series, bins=10))
print(event_series[15:40], predictions[15:40])

# Save data
with open("l2_classifiers.dat", "w") as file:
    pickle.dump({
            "observation": observation, 
            "transition": transition,
            "type_classifier": typeClassifier, 
            "signal_classifiers": signalClassifiers
        }, file)