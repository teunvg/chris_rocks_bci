from buffer_bci import preproc, bufhelp
import pickle
import time
import socket
import numpy as np

ftc, hdr = bufhelp.connect()

trlen_ms = 1100
sample_ms = 250

# Load recorded data
with open('l2_classifiers' + '.dat', 'r') as file:
    raw = pickle.load(file)
    classifier, hmm, feature = raw['classifier'], raw['hmm'], raw['feature']

# Connect to Cybathlon
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error:
    print 'Failed to create socket'
    sys.exit()

host = 'localhost'
port = 5555
connection = (s, (host, port))

def sendCommand(connection, player, cmd):
    cmds = ['SPEED', 'JUMP', 'ROLL']
    msg = player * 10 + cmds.index(cmd)
    connection[0].sendto(ord(cmd), connection[1])

# Model parameters
commands = ['IDLE', 'SPEED', 'ROLL', 'JUMP']
player = 1

window_size = 64
hmm_window = 30
hmm_step = 8

# Fetch nummer of channels and sampling rate
n_channels = hdr.nChannels
sampling_rate = hdr.fSample
sample_window = trlen_ms * sampling_rate / 1000.
sample_rate = sample_ms * sampling_rate / 1000.

# Start recording loop
observations = [0] * 30
ts = 0
cur_sample, n_events = ftc.poll()
prev_events = n_events
print("Start of recording")
while True:
    timespan = (cur_sample - sample_window, cur_sample - 1)
    eventspan = (prev_events, n_events - 1)
    datum = ftc.getData(timespan)
    event = ftc.getEvents(eventspan)

    # Preprocess data
    datum = datum[-window_size:,:n_channels]
    datum = preproc.detrend([datum])[0]
    datum = preproc.spatialfilter([datum], type="car")[0]

    # Take fft
    subwindow = window_size
    datum = np.abs(np.fft.rfft(1 * datum, axis=0))
    print(datum.shape)
    freqs = np.array([float(i)/subwindow*sampling_rate for i in range(subwindow/2+1)])
    deltas = [np.logical_and(freqs > i, freqs < j) for i, j in zip([0.4], [4])]
    mus = [np.logical_and(freqs > i, freqs < j) for i, j in zip([7], [14])]
    betas = [np.logical_and(freqs > i, freqs < j) for i, j in zip([14, 19, 26], [19, 26, 30])]
    bands = deltas + mus + betas
    datum = np.array([datum[band, :].mean(axis=0).tolist() for band in bands])

    # Normalise
    flat_data = datum.flatten()
    demeaned_data = flat_data - feature['means']
    norm_data = demeaned_data / feature['stds']
    datum = datum.reshape(len(bands), -1)

    # Run classification
    observation = classifier.predict(datum.reshape(1, -1))[0]
    observations.append(observation)

    if ts % hmm_step == 0:
        part_pred = HMM.predict(np.array(observations[-hmm_window:]))[-hmm_step:].max()
        pred = part_pred - 6 if part_pred > 6 else 0

        action = commands[pred]
        if action != 'IDLE':
            sendCommand(connection, player, action)

    # Wait for next block
    ts += 1
    prev_events = n_events
    cur_sample, n_events = ftc.wait(cur_sample + sample_rate, -1, sample_ms)
    if any([ev.type == 'stimulus.training' and ev.value == 'end' for ev in events[-1]]):
        break

# Save data
with open("live_subject_data.dat", "w") as file:
    pickle.dump({"hdr": hdr, "events": events, "data": data}, file)
print("End of recording")