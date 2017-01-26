import numpy as np
from scipy import signal
import seaborn as sns
from buffer_bci import preproc, linear
import pickle
import pywt as pywt
import sklearn as sk

# Load recorded data
events = data = []
with open('live_subject_data' + '.dat', 'r') as file:
    raw = pickle.load(file)
    hdr, data, events = raw['hdr'], raw['data'], raw['events']

n_channels = 10
freq = 250.

window_size = 256
window_shift = 3

ix = np.where([len(event) > 0 for event in events])[0]
gap_ix = ix[np.diff(ix) > 30]
deckCut = gap_ix[3] + 1

# Preprocessing hooray!
data = [datum[-window_size:,:n_channels] for i, datum in enumerate(data)]
data = preproc.detrend(data)
#data, badch = preproc.badchannelremoval(data)
#for ch in badch:
#    del channels[ch]
data = preproc.spatialfilter(data, type="car")
#data, events, badtrials = preproc.badtrialremoval(data, events)

# Take fft
expfun = (1-0.5*(1/np.exp(np.linspace(0, 4.2, window_size)))).reshape((window_size,1))
data = [np.abs(np.fft.rfft(1 * datum, axis=0)) for datum in data]
freqs = np.array([float(i)/window_size*freq for i in range(window_size/2+1)])
deltas = [np.logical_and(freqs > i, freqs < j) for i, j in zip([0.4], [4])]
mus = [np.logical_and(freqs > i, freqs < j) for i, j in zip([7], [14])]
betas = [np.logical_and(freqs > i, freqs < j) for i, j in zip([14, 19, 25], [19, 25, 30])]
bands = deltas + mus + betas
data = [np.array([datum[band, :].mean(axis=0).tolist() for band in bands]) for datum in data]

# Normalise
flat_data = np.array([datum.flatten().tolist() for datum in data])
feature_means = flat_data.mean(axis=0)
demeaned_data = flat_data - feature_means
feature_stds = demeaned_data.std(axis=0)
norm_data = demeaned_data / feature_stds
data = [np.array(datum).reshape(len(bands), -1) for datum in norm_data.tolist()]

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
event_series = np.cumsum(event_types)[:-window_shift]
data = data[window_shift:]

# Split dah sets
train_data, train_events = data[:deckCut], event_series[:deckCut]
test_data, test_events = data[deckCut+30:], event_series[deckCut+30:]

# CLASSIFICATION :D :D :D
def fit(data, events, C=5, kernel='rbf'):
    event_types = list(set(events))
    events = np.array(events)
    lolweights = {i: (events == i).mean() for i in event_types}
    classifier = sk.svm.SVC(C=C, kernel=kernel, probability=False)
    classifier.fit(np.array([datum.flatten().tolist() for datum in data]), events)
    return classifier

classifier = fit(train_data, train_events)

all, correct = 0, 0
for datum, event in zip(train_data, train_events):
    pred = classifier.predict(datum.reshape(1, -1))[0]
    correct += int(pred == event) if event > 0 else 0
    all += 1 if event > 0 else 0

print(float(correct) / all)

all, correct = 0, 0
hits = np.zeros((10,))
confusion = np.zeros((10, 10))
for datum, event in zip(test_data, test_events):
    pred = classifier.predict(datum.reshape(1, -1))[0]
    confusion[pred, event] += 1
    hits[pred] += 1
    correct += int(pred == event) if event > 0 else 0
    all += 1 if event > 0 else 0

print(np.histogram(event_series, bins=10)[0])
print(hits)
print(float(correct) / all)

mean_confusion = confusion.sum(axis=1)
norm_confusion = np.divide(confusion, mean_confusion.reshape(-1,1))
print(norm_confusion)

# Save data
with open("l2_classifiers.dat", "w") as file:
    pickle.dump({"observation": norm_confusion, "classifier": classifier}, file)
print("End of recording")