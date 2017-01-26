from buffer_bci import preproc, bufhelp
import pickle
import time

ftc, hdr = bufhelp.connect()

trlen_ms = 1100
sample_ms = 250

# Fetch nummer of channels and sampling rate
n_channels = 10 #hdr.nChannels
sampling_rate = hdr.fSample
sample_window = trlen_ms * sampling_rate / 1000.
sample_rate = sample_ms * sampling_rate / 1000.

# Start recording loop
data, events = [], []
cur_sample, n_events = ftc.poll()
print('Waiting for start event...')
cur_sample, n_events = ftc.wait(-1, n_events, 1000000)
prev_events = n_events
print("Start of recording")
while True:
    timespan = (cur_sample - sample_window, cur_sample - 1)
    eventspan = (prev_events, n_events - 1)
    data.append(ftc.getData(timespan))
    events.append(ftc.getEvents(eventspan))
    prev_events = n_events
    cur_sample, n_events = ftc.wait(cur_sample + sample_rate, -1, sample_ms)
    if any([ev.type == 'stimulus.training' and ev.value == 'end' for ev in events[-1]]):
        break

# Save data
with open("live_subject_data.dat", "w") as file:
    pickle.dump({"hdr": hdr, "events": events, "data": data}, file)
print("End of recording")