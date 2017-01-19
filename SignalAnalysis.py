import numpy as np
from scipy import signal
import seaborn as sns
from buffer_bci import preproc, linear
import pickle
import pywt as pywt

def getspectralfilter(type, band, rate):
    b, a = signal.butter(6, [2 * x / float(rate) for x in band], type)
    return (b, a)

events = data = []
with open('subject_chris_actual' + '.dat', 'r') as file:
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

data = [datum[:,:n_channels] for datum in data]
data = preproc.detrend(data)
data, badch = preproc.badchannelremoval(data)
for ch in badch:
    del channels[ch]
data = preproc.spatialfilter(data, type="car")
data = applyfilter(power_50hz, data)
#data = [signal.filtfilt(*(bandstop1 + (datum,)), axis=0) for datum in data]
#data = [signal.filtfilt(*(bandstop2 + (datum,)), axis=0) for datum in data]
#data = preproc.spectralfilter(data, (0, .1, 30, 38), freq)
data, events, badtrials = preproc.badtrialremoval(data, events)
print(badtrials)

types, prepData, startData, stopData = [], [], [], []
for event, datum in zip(events, data):
    if event.type == 'stimulus.prepare':
        prepData.append(datum)
        types.append(event.value)
    elif event.type == 'stimulus.start':
        startData.append(datum)
    elif event.type == 'stimulus.stop':
        stopData.append(datum)

getType = lambda type, data: [d for t, d in zip(types, data) if t == type]
splitData = lambda classes, data: {c: np.array(getType(c, data)) for c in classes}

classes = set(types)
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

'''
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
classes = list(classes)
l_events = np.array([classes.index(type) for type in types])
classifier = {
    'prep': linear.fit(applyfilter(LRPband, prepData), l_events),
    'start': linear.fit(applyfilter(ERDband, startData), l_events),
    'stop': linear.fit(applyfilter(ERSband, stopData), l_events)
}

print(classes)
for event, datum in zip(events, data):
    print(event.type, event.value, datum)
    for t, c in classifier.items():
        pred = linear.predict(c, [datum])
        print(t, pred)
    break

sns.plt.show()