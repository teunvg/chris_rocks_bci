import numpy as np
from scipy import signal
import seaborn as sns
from buffer_bci import preproc
import pickle
import pywt as pywt

def getspectralfilter(type, band, rate):
    b, a = signal.butter(5, [2 * x / float(rate) for x in band], type)
    return (b, a)

events = data = []
with open('chris_data_2' + '.dat', 'r') as file:
    data, events = pickle.load(file).values()

channels = [''] * 32
with open('channels.csv', 'r') as file:
    file.readline()
    for line in file:
        ch_num, ch_name, x, y = line.split(',')
        channels[int(ch_num) - 1] = ch_name

freq = 250.

bandpass = getspectralfilter('bandpass', [.4, 38], freq)
bandstop = getspectralfilter('bandstop', [45, 55], freq)

data = [datum[:,:32] for datum in data]
data = preproc.detrend(data)
data, badch = preproc.badchannelremoval(data)
for ch in badch:
    del channels[ch]
data = preproc.spatialfilter(data, type="car")
data = [signal.filtfilt(*(bandpass + (datum,)), axis=0) for datum in data]
data = [signal.filtfilt(*(bandstop + (datum,)), axis=0) for datum in data]
#data = preproc.spectralfilter(data, (0, .1, 30, 38), freq)
#data, events, badtrials = preproc.badtrialremoval(data, events)

types, startData, endData = [], [], []
for event, datum in zip(events, data):
    if event.type == 'stimulus.tgtFlash':
        startData.append(datum)
        types.append(event.value)
    elif event.type == 'stimulus.tgtHide' and len(startData) == len(endData) + 1:
        endData.append(datum)

getType = lambda type, data: [d for t, d in zip(types, data) if t == type]
splitData = lambda classes, data: {c: np.array(getType(c, data)) for c in classes}

classes = set(types)
timeData = {
    'start': splitData(classes, startData),
    'end': splitData(classes, endData)
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
    fig, axs = sns.plt.subplots(4, 8, sharey='all', sharex='all')
    for i, ax in enumerate(axs.flatten()):
        if i < len(channels):
            ax.set_title(channels[i])
            for eventType, eventMeans in typeMeans.items():
                ax.plot(eventMeans[:,i], label=eventType)
                #sns.tsplot(data=eventMeans[:,:,i], ci=[68, 95], ax=ax, color=colors[eventType])
    sns.plt.legend()

specData = treeLambda(timeData, lambda datum: signal.spectrogram(datum, freq, nperseg=64, noverlap=63, axis=1))
#waveData = treeLambda(timeData, lambda datum: pywt.wavedec(datum, 'db1', axis=1))
specMeans = treeLambda(specData, lambda datum: (datum[0], datum[1], np.mean(datum[2], axis=0)))
for eventType, typeMeans in specMeans.items():
    for eventClass, eventMeans in typeMeans.items():
        fig, axs = sns.plt.subplots(4, 8, sharey='all', sharex='all')
        print(eventType + ' for ' + eventClass)
        for i, ax in enumerate(axs.flatten()):
            if i < len(channels):
                ax.set_title(channels[i])
                x, y = np.meshgrid(*reversed(eventMeans[0:2]))
                ax.contourf(x, y, np.multiply(eventMeans[2][:,i,:], np.matrix([[f ** 2 for f in eventMeans[0]]] * len(eventMeans[1])).T))
        ax.set_ylim([0, 45])
sns.plt.show()