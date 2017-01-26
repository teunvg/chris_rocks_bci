import numpy as np
from scipy import signal
import seaborn as sns
from buffer_bci import preproc, linear
import pickle
import pywt as pywt
import sklearn as sk
from hmmlearn import hmm

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
for i in range(gap_ix.size):
    gap = gap_ix[i]
    data[gap+5:gap+35] = []
    events[gap+5:gap+35] = []
    gap_ix[i+1:] -= 30
l1Cut, l2Cut = gap_ix[1] + 1, gap_ix[1] + 1

# Preprocessing hooray!
data = [datum[-window_size:,:n_channels] for i, datum in enumerate(data)]
data = preproc.detrend(data)
#data, badch = preproc.badchannelremoval(data)
#for ch in badch:
#    del channels[ch]
data = preproc.spatialfilter(data, type="car")
#data, events, badtrials = preproc.badtrialremoval(data, events)

# Take fft
subwindow = window_size
expfun = (1-0.5*(1/np.exp(np.linspace(0, 4.2, window_size)))).reshape((window_size,1))
data = [np.abs(np.fft.rfft(1 * datum.reshape(subwindow, -1), axis=0)) for datum in data]
freqs = np.array([float(i)/subwindow*freq for i in range(subwindow/2+1)])
deltas = [np.logical_and(freqs > i, freqs < j) for i, j in zip([0.4], [4])]
mus = [np.logical_and(freqs > i, freqs < j) for i, j in zip([7], [14])]
betas = [np.logical_and(freqs > i, freqs < j) for i, j in zip([14, 19, 26], [19, 26, 30])]
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
event_series = np.cumsum(event_types)

#event_series = np.ceil(event_series / 3)

if window_shift > 0:
    event_series = event_series[:-window_shift]
    data = data[window_shift:]

# Split dah sets
train_data, train_events = data[:l1Cut], event_series[:l1Cut]
hmm_data, hmm_events = data[l1Cut:l2Cut], event_series[l1Cut:l2Cut]
test_data, test_events = data[l2Cut:], event_series[l2Cut:]

# CLASSIFICATION :D :D :D
def fit(data, events, C=4, kernel='rbf'):
    event_types = list(set(events))
    events = np.array(events)
    lolweights = {i: (events == i).mean() for i in event_types}
    classifier = sk.svm.SVC(C=C, kernel=kernel, probability=True)
    classifier.fit(np.array([datum.flatten().tolist() for datum in data]), events)
    return classifier

order = np.argsort(train_events)
train_data = np.array(train_data)[order]
train_events = np.array(train_events)[order]
classifier = fit(train_data, train_events)

all, correct = 0, 0
for datum, event in zip(train_data, train_events):
    pred = classifier.predict_proba(datum.reshape(1, -1)).argmax()
    correct += int(pred == event) if event > 0 else 0
    all += 1 if event > 0 else 0

print(float(correct) / all)

all, correct = 0, 0
hits, tots = np.zeros((10,)), np.zeros((10,))
confusion = np.ones((10, 10)) * .2
for datum, event in zip(test_data, test_events):
    pred = classifier.predict_proba(datum.reshape(1, -1)).argmax()
    confusion[pred, event] += 1
    hits[pred] += int(pred == event)
    tots[pred] += 1
    correct += int(pred == event) if event > 0 else 0
    all += 1 if event > 0 else 0

print(np.histogram(event_series, bins=10)[0])
print(hits / tots)
print(float(correct) / all)

mean_confusion = confusion.sum(axis=1)
norm_confusion = np.divide(confusion, mean_confusion.reshape(-1,1))
print(norm_confusion)

# Save data
with open("l1_classifiers.dat", "w") as file:
    pickle.dump({
        "observation": norm_confusion, 
        "classifier": classifier,
        "feature": {
            "means": feature_means,
            "stds": feature_stds
        }
        }, file)
print("End of l1 training")

# The second layer will be majestic and reach 30%
observations = [classifier.predict_proba(datum.reshape(1, -1)).argmax() for datum in test_data]

emissionMatrix = np.matrix(norm_confusion)
transitionMatrix = np.matrix([
    [.7, .1, .1, .1, 0, 0, 0, 0, 0, 0],
    [0, .56, 0, 0, .44, 0, 0, 0, 0, 0],
    [0, 0, .56, 0, 0, .44, 0, 0, 0, 0],
    [0, 0, 0, .56, 0, 0, .44, 0, 0, 0],
    [0, 0, 0, 0, .6, 0, 0, .4, 0, 0],
    [0, 0, 0, 0, 0, .6, 0, 0, .4, 0],
    [0, 0, 0, 0, 0, 0, .6, 0, 0, .4],
    [.44, 0, 0, 0, 0, 0, 0, .56, 0, 0],
    [.44, 0, 0, 0, 0, 0, 0, 0, .56, 0],
    [.44, 0, 0, 0, 0, 0, 0, 0, 0, .56]
    ])

HMM = hmm.MultinomialHMM(n_components=10)
HMM.transmat_ = transitionMatrix
HMM.startprob_ = np.array([.304] + [.077] * 3 + [.086] * 3 + [.069] * 3)
HMM.emissionprob_ = emissionMatrix

hmm_window = 30
samples = len(observations) - hmm_window
Xix = np.array([observations[i:i+hmm_window] for i in range(samples)]).flatten()
X = np.zeros((len(Xix), 10))
for i, observation in enumerate(Xix):
    X[i,observation] = 1
Xlengths = np.array([hmm_window] * samples)

#HMM.fit(np.array(hmm_events).reshape(-1,1))
#print(test_events[20:40], observations[20:40], HMM.predict(np.array(observations)[20:40]))

hmm_step = 12

part_pred = [HMM.predict(np.array(observations)[i:i+hmm_window])[-hmm_step:].max() for i in range(0, len(test_events)-hmm_window, hmm_step)]
pred = np.array([p - 6 if p > 6 else 0 for p in part_pred])

part_true = [test_events[i+hmm_window-hmm_step:i+hmm_window].max() for i in range(0, len(test_events)-hmm_window, hmm_step)]
true = np.array([p - 6 if p > 6 else 0 for p in part_true])

print(pred, true, float(np.logical_and(pred == true, true > 0).sum()) / (true > 0).sum())
print(float(np.logical_and(pred == true, true == 0).sum()) / (true == 0).sum())

# Save data
with open("l2_classifiers.dat", "w") as file:
    pickle.dump({
        "hmm": HMM, 
        "classifier": classifier,
        "feature": {
            "means": feature_means,
            "stds": feature_stds
        }
        }, file)
print("End of l2 training")

print("done")

# 1/3 * 1/8 * 1/8 = ~0.5% chance of random perfect sequence
# Mathematically (back of the envelope) every 5 minutes for time window of 1.5s
# This does not take into account deliberate disturbances (since we are doing a task)
# Conclusion: Idle can quite safely be left out as an observation

# Given that we can already classify the class of any given brain signature with close
# to 99% certainty over a sampling time of 1.5s with 250ms intervals, a 5+/7 check (~1% error rate)
# every 3s might already perform nearly impeccably. (this also comes down to once every 5 minutes) [Binomial + simple repetition]
# 40% performance with 3+/7 check: ~11% false positive rate, ~58% true positive rate [2.5 errors every 75 seconds, correct 15 times]
# 50% performance with 3+/7 check: ~11% false positive rate, ~77% true positive rate [2.5 errors every 75 seconds, correct 20 times]
# 60% performance with 4+/7 check: ~4% false positive rate, ~71% true positive rate [1 error every 75 seconds, correct 17.5 times]
# 70% performance with 4+/7 check: ~4% false positive rate, ~87% true positive rate [1 error every 75 seconds, correct 22 times]
# 80% performance with 4+/7 check: ~4% false positive rate, ~97% true positive rate [1 error every 75 seconds, correct 24 times]
# 90% performance with 5+/7 check: ~1% false positive rate, ~97% true positive rate [practically flawless]
# 98% performance with 6+/7 check: ~.4% false positive rate, ~99% true positive rate [practically flawless]
# Theoretically, a good implementation of the HMM should only improve this classification,
# so that a performance should be aimed at at least the 65% level, including idleness, for any meaningful improvement.

# ToDo x: Save confusion matrix and classifiers
#      x: Change second training phase to be less strictly timed and record more data (every 250ms), but with full instructions
#      x: Record data from second training phase
#---------Until here on Monday morning
#      4: Train HMM using initial confusion matrix with the EM algorithm (short procedure)
#      5: Test trained HMM on separate testing dataset by using the Virterbi algorithm
#      6: Implement alternative second-layer procedures and compare 
#      7: Implement live feedback procedure for Cybathlon and improve training
#---------Until here on Monday evening
#      8: Test Cybathlon performance
#      9: Tweak training procedure(s) accordingly
#---------Done by Tuesday afternoon