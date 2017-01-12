from psychopy import visual, core
from buffer_bci import bufhelp
from random import randint, uniform, shuffle
import numpy as np

# experiment settings
training_blocks = 3
training_stims = 10

prep_length = (.6, 1.2)
stim_length = (1., 2.)
interstim_time = (.75, 1.5)
break_time = 15

# set up the environment
stim_dist = 5
stim_radius = 1

ftc, hdr = bufhelp.connect()

screen = visual.Window([960,540], monitor="testMonitor", units="deg", fullscr=False)

# create the stimuli
text = visual.TextStim(win=screen, text="Please wait...", color=(1,1,1))
fixation = visual.GratingStim(win=screen, size=0.5, pos=[0,0], sf=0)

labelText = ['F', 'LH', 'RH']

stimN = len(labelText)
circles = [None] * stimN
labels = [None] * stimN
for i, label in enumerate(labelText):
    angle = np.pi + i * 2 * np.pi / stimN
    position = (np.sin(angle) * stim_dist, np.cos(angle) * stim_dist)
    circles[i] = visual.Circle(win=screen, radius=stim_radius, pos=position, fillColor=(.5, .5, .5))
    labels[i] = visual.TextStim(win=screen, text=label, pos=position, color=(1,1,1))

def setCircle(circle, color, type, label):
    circle.setFillColor(color)
    bufhelp.sendEvent("stimulus." + type, label)

def showElements(screen, elts, time=1):
    for el in elts:
        el.draw()
    screen.update()
    core.wait(time)

def showElement(screen, el, time=1):
    showElements(screen, [el], time)

# start calibration 
bufhelp.sendEvent("startPhase.cmd", "calibration")
text.setText("Prepare for training")
showElement(screen, text)
showElements(screen, circles + labels + [fixation], 2)

stimuli = list(range(stimN)) * training_stims;

colors = ((0, 0, .5), (0, .5, 0), (.5, .5, .5))
states = ("prepare", "start", "stop")
timings = (prep_length, stim_length, interstim_time)

bufhelp.sendEvent("stimulus.training", "start")
for b in range(training_blocks):
    shuffle(stimuli)

    for stim in stimuli:
        for color, state, timing in zip(colors, states, timings):
            setCircle(circles[stim], color, state, labelText[stim])
            showElements(screen, circles + labels + [fixation], uniform(*timing))

    if b < training_blocks - 1:
        text.setText("You have a %ds break now" % break_time)
        showElements(screen, circles + labels + [text], break_time)
        showElements(screen, circles + labels + [fixation], 1)

# wait for training to complete
text.setText("Calibration phase is over, please wait")
bufhelp.sendEvent("stimulus.training", "end")
showElement(screen, text, 5)

# start testing phase


# end of experiment