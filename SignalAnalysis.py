import numpy as np
from scipy import signal
import seaborn as sns
from buffer_bci import preproc, linear
import pickle
import pywt as pywt
import sklearn as sk

def getspectralfilter(type, band, rate):
    b, a = signal.butter(6, [2 * x / float(rate) for x in band], type)
    return (b, a)

events = data = []
with open('teun_im' + '.dat', 'r') as file:
    raw = pickle.load(file)
    hdr, data, events = raw['hdr'], raw['data'], raw['events']

n_channels = 10
freq = 250.

channels = [''] * n_channels
with open('channels10.csv', 'r') as file:
    file.readline()
    for line in file:
        ch_num, ch_name, x, y = line.split(',')
        channels[int(ch_num) - 1] = ch_name

allband = getspectralfilter('bandpass', [.4, 38], freq)
LRPband = getspectralfilter('bandpass', [.4, 3.5], freq)
ERDband = getspectralfilter('bandpass', [8, 38], freq)
ERSband = getspectralfilter('bandpass', [14, 38], freq)
power_50hz = getspectralfilter('bandstop', [45, 55], freq)

applyfilter = lambda filter, data: [signal.filtfilt(*(filter + (datum,)), axis=0) for datum in data]

data = [datum[:,:n_channels] for i, datum in enumerate(data) if np.floor(i / 90) not in [1]]
events = [event for i, event in enumerate(events) if np.floor(i / 90) not in [1]]
data = preproc.detrend(data)
#data, badch = preproc.badchannelremoval(data)
#for ch in badch:
#    del channels[ch]
data = preproc.spatialfilter(data, type="car")
data = applyfilter(power_50hz, data)
#data = [signal.filtfilt(*(bandstop1 + (datum,)), axis=0) for datum in data]
#data = [signal.filtfilt(*(bandstop2 + (datum,)), axis=0) for datum in data]
#data = preproc.spectralfilter(data, (0, .1, 30, 38), freq)
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

#order = np.argsort([event.sample for event in events])
#events, data = zip(*[(events[i], data[i]) for i in order])
types, prepData, startData, stopData = [], [], [], []
for event, datum in zip(events, data):
    if event.type == 'stimulus.prepare':
        prepData.append(datum)
        types.append(event.value)
    elif event.type == 'stimulus.start':
        startData.append(datum)
    elif event.type == 'stimulus.stop':
        stopData.append(datum)

print(types)

getType = lambda type, data: [d for t, d in zip(types, data) if t == type]
splitData = lambda classes, data: {c: np.array(getType(c, data)) for c in classes}

classes = set(types)
'''
timeData = {
    'prep': splitData(classes, applyfilter(LRPband, prepData)),
    'start': splitData(classes, applyfilter(ERDband, startData)),
    'stop': splitData(classes, applyfilter(ERSband, stopData))
}

treeLambda = lambda data, fun: {k1: {k2: fun(data[k1][k2]) for k2 in data[k1].keys()} for k1 in data.keys()}
treeMeans = lambda data: treeLambda(data, lambda datum: np.mean(datum, axis=0))

colors = {
    'LH': 'b',
    'RH': 'g',
    'F': 'r'
}

timeMeans = treeMeans(timeData)
for typeMeans in timeMeans.values():
    fig, axs = sns.plt.subplots(2, 5, sharey='all', sharex='all')
    for i, ax in enumerate(axs.flatten()):
        if i < len(channels):
            ax.set_title(channels[i])
            #print(typeMeans)
            for eventType, eventMeans in typeMeans.items():
                ax.plot(eventMeans[:,i], label=eventType)
                #sns.tsplot(data=eventMeans[:,:,i], ci=[68, 95], ax=ax, color=colors[eventType])
    sns.plt.legend()

specData = treeLambda(timeData, lambda datum: signal.spectrogram(datum, freq, nperseg=64, noverlap=63, axis=1))
#waveData = treeLambda(timeData, lambda datum: pywt.wavedec(datum, 'db1', axis=1))
specMeans = treeLambda(specData, lambda datum: (datum[0], datum[1], np.mean(datum[2], axis=0)))
for eventType, typeMeans in specMeans.items():
    for eventClass, eventMeans in typeMeans.items():
        fig, axs = sns.plt.subplots(2, 5, sharey='all', sharex='all')
        print(eventType + ' for ' + eventClass)
        for i, ax in enumerate(axs.flatten()):
            if i < len(channels):
                ax.set_title(channels[i])
                x, y = np.meshgrid(*reversed(eventMeans[0:2]))
                ax.contourf(x, y, np.multiply(eventMeans[2][:,i,:], np.matrix([[f ** 2 for f in eventMeans[0]]] * len(eventMeans[1])).T))
        ax.set_ylim([0, 45])
'''

# CLASSIFICATION :D :D :D
def fit(data, events, C=1, kernel='rbf'):
    classifier = sk.svm.SVC(C=C, kernel=kernel, probability=True)
    classifier.fit(np.array([datum.flatten().tolist() for datum in data]), np.array(events))
    return classifier

TypeFilter = lambda data: [datum[:,:,0:31:5] for datum in data]
LRPfilter = lambda data: [datum[0,:,0:6] for datum in data]
ERDfilter = lambda data: [datum[1:,:,0:31:5] for datum in data]
ERSfilter = lambda data: [datum[2,:,0:31:5] for datum in data]

#ev_types = list(set([event.type for event in events])) 
#classes = list(classes)
#print(ev_types, classes)
classes = ["RH", "LH", "F"]
ev_types = ["stimulus.prepare", "stimulus.start", "stimulus.stop"]
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
        '''
        l_events = np.array([classes.index(type) for j, type in enumerate(types) if j != i])
        all_events = l_events.tolist() + (l_events + 3).tolist() + (l_events + 6).tolist()
        classifier = fit(startPart + stopPart + prepPart, all_events)

        prediction_matrix = lambda datum: classifier.predict_proba(datum.reshape(1, -1))
        '''
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

sns.plt.show()