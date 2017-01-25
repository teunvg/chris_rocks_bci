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
print((1-(1/np.exp(np.linspace(0, 4.2, window_size)))).reshape((window_size,1)).T)
data = [np.abs(np.fft.rfft(1 * datum, axis=0)) for datum in data]
freqs = np.array([float(i)/window_size*freq for i in range(window_size/2+1)])
deltas = [np.logical_and(freqs > i, freqs < j) for i, j in zip([0.4], [4])]
mus = [np.logical_and(freqs > i, freqs < j) for i, j in zip([7], [14])]
betas = [np.logical_and(freqs > i, freqs < j) for i, j in zip([14, 22], [22, 30])]
data = [np.array([datum[band, :].mean(axis=0).tolist() for band in deltas + mus + betas]) for datum in data]

# Normalise
flat_data = np.array([datum.flatten().tolist() for datum in data])
feature_means = flat_data.mean(axis=0)
demeaned_data = flat_data - feature_means
feature_stds = demeaned_data.std(axis=0)
norm_data = demeaned_data / feature_stds
#data = [np.array(datum).reshape(4, -1) for datum in norm_data.tolist()]

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
def fit(data, events, C=.9, kernel='rbf'):
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
print(confusion)

'''
all = 0
correct = np.zeros((9,))
total = np.zeros((9,))
classcor = np.zeros((3,))
classtot = np.zeros((3,))
typecor = np.zeros((3,))
typetot = np.zeros((3,))
confusion = np.zeros((10,9))
confusion[:,:] = 1.

for n in range(1):
    for i in range(len([1])):
        prepPart, startPart, stopPart = prepData[:i] + prepData[(i+1):], startData[:i] + startData[(i+1):], stopData[:i] + stopData[(i+1):]
        
        t_events = (0,) * len(prepPart) + (1,) * len(startPart) + (2,) * len(stopPart)
        classifier = fit(TypeFilter(prepPart + startPart + stopPart), t_events, C=1)

        l_events = [classes.index(type) for j, type in enumerate(types) if j != i]
        l_events, prepPart, startPart, stopPart = zip(*[(l_events[j], prepPart[j], startPart[j], stopPart[j]) for j in np.argsort(l_events)])
        classifiers = (
            ('LRP', LRPfilter, fit(LRPfilter(prepPart), l_events, C=5)),
            ('ERD', ERDfilter, fit(ERDfilter(startPart), l_events, C=5)),
            ('ERS', ERSfilter, fit(ERSfilter(stopPart), l_events, C=5))
        )

        prediction_matrix = lambda datum: (np.array([(c.predict_proba(f([datum])[0].reshape(1, -1))).flatten().tolist() for t, f, c in classifiers]) \
                                * classifier.predict_proba(TypeFilter([datum])[0].reshape(1, -1)).T).flatten()
        
        l_events = np.array([classes.index(type) for j, type in enumerate(types) if j != i])
        all_events = l_events.tolist() + (l_events + 3).tolist() + (l_events + 6).tolist()
        classifier = fit(startPart + stopPart + prepPart, all_events)

        prediction_matrix = lambda datum: classifier.predict_proba(datum.reshape(1, -1))
        
        print('bootstrap', i)
        cheat = 0
        if (i+cheat < len(types)):
            for ev_type, datum in zip([0, 1, 2], [prepData[i+cheat], startData[i+cheat], stopData[i+cheat]]):
                #ev_type = ev_types.index(event.type)
                ev_class = classes.index(types[i+cheat])  #classes.index(event.value)
                ix = ev_type * 3 + ev_class
                print(ix, ev_type, ev_class)
                #pred = prediction_matrix(datum)
                #predmat = pred.reshape(3,3)
                pred_type = classifier.predict(TypeFilter([datum])[0].reshape(1, -1)) #predmat.sum(axis=1).argmax()
                t, f, c = classifiers[pred_type]
                pred_class = c.predict(f([datum])[0].reshape(1, -1)) #predmat.sum(axis=0).argmax()
                #preds = pred.argpartition(-3)[-3:]
                predix = pred_type * 3 + pred_class #pred.argmax()
                hit = ix == predix #(ev_type, ev_class) == (pred_type, pred_class) #
                confusion[ix, predix] += 1
                all += 1
                total[ix] += 1
                if hit:
                    correct[ix] += 1
                classtot[ev_class] += 1
                if ev_class == pred_class:
                    classcor[ev_class] += 1
                typetot[ev_type] += 1
                if ev_type == pred_type:
                    typecor[ev_type] += 1
                #print(ix, pred.argmax())
                #print((ev_type, ev_class), (pred_type, pred_class))

rate = np.divide(correct, total)
print(rate, np.mean(rate))

classrate = np.divide(classcor, classtot)
print(classrate, np.mean(classrate))

typerate = np.divide(typecor, typetot)
print(typerate, np.mean(typerate))

mean_confusion = confusion.sum(axis=1)
norm_confusion = np.divide(confusion, mean_confusion.reshape(-1,1))
print(norm_confusion)

# Rerun unbootstrapped
t_events = (0,) * len(prepData) + (1,) * len(startData) + (2,) * len(stopData)
classifier = fit(TypeFilter(prepData + startData + stopData), t_events)

l_events = [classes.index(type) for j, type in enumerate(types)]
l_events, prepPart, startPart, stopPart = zip(*[(l_events[j], prepData[j], startData[j], stopData[j]) for j in np.argsort(l_events)])
classifiers = {
    'ERD': fit(ERDfilter(prepPart), l_events, C=15),
    'ERS': fit(ERSfilter(startPart), l_events, C=15),
    'LRP': fit(LRPfilter(stopPart), l_events, C=15)
}

# Save data
with open("l1_classifiers.dat", "w") as file:
    pickle.dump({"observation": norm_confusion, "type_classifier": classifier, "signal_classifiers": classifiers}, file)
print("End of recording")

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
'''