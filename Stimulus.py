from psychopy import visual, core
from buffer_bci import bufhelp
from random import randint, uniform, shuffle
import numpy as np

# experiment settings
training_blocks = 3
training_stims = 10

stim_length = (1.5, 2.5)
interstim_time = (1, 1.5)
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

def selectCircle(circle, label):
    #selected = randint(0, len(circles) - 1)
    circle.setFillColor((0, .5, 0))
    bufhelp.sendEvent("stimulus.tgtFlash", label)

def deselectCircles(circles):
    for circle in circles:
        circle.setFillColor((.5, .5, .5))
    bufhelp.sendEvent("stimulus.tgtHide", True)

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

bufhelp.sendEvent("stimulus.training", "start")
for b in range(training_blocks):
    shuffle(stimuli)

    for stim in stimuli:
        selectCircle(circles[stim], labelText[stim])
        showElements(screen, circles + labels + [fixation], uniform(*stim_length))
        deselectCircles(circles)
        showElements(screen, circles + labels + [fixation], uniform(*interstim_time))

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