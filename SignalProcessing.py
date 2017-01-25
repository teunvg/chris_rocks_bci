#!/usr/bin/python

from buffer_bci import preproc, bufhelp
import pickle

ftc, hdr = bufhelp.connect()

trlen_ms = 1500
run = True

print("Waiting for startPhase.cmd event.")
while run:
    e = bufhelp.waitforevent("startPhase.cmd",1000, False)
    
    if e is not None:

        if e.value == "calibration":
            print("Calibration phase")
            data, events, stopevents = bufhelp.gatherdata(["stimulus.prepare", "stimulus.start", "stimulus.stop"], trlen_ms, ("stimulus.training", "end"), milliseconds=True)
            with open("subject_data.dat", "w") as file:
                pickle.dump({"hdr": hdr, "events": events, "data": data}, file)
            print("End calibration phase")

        elif e.value == "train":
            print("Training classifier")
            data = preproc.detrend(data)
            data, badch = preproc.badchannelremoval(data)
            data = preproc.spatialfilter(data)
            data = preproc.spectralfilter(data, (0, .1, 10, 12), bufhelp.fSample)
            data, events, badtrials = preproc.badtrialremoval(data, events)
            mapping = {('stimulus.tgtFlash', '0'): 0, ('stimulus.tgtFlash', '1'): 1}
            linear.fit(data,events,mapping)
            bufhelp.update()
            bufhelp.sendevent("sigproc.training","done")

        elif e.value =="testing":
            print("Feedback phase")
            while True:
                data, events, stopevents = bufhelp.gatherdata(["stimulus.columnFlash","stimulus.rowFlash"],trlen_ms,[("stimulus.feedback","end"), ("stimulus.sequence","end")], milliseconds=True)

                if isinstance(stopevents, list):
                    if any(map(lambda x: "stimulus.feedback" in x.type, stopevents)):
                        break
                else:
                    if "stimulus.feedback" in stopevents.type:
                        break
                
                data = preproc.detrend(data)
                data, badch = preproc.badchannelremoval(data)
                data = preproc.spatialfilter(data)
                data = preproc.spectralfilter(data, (0, .1, 10, 12), bufhelp.fSample)
                data2, events, badtrials = preproc.badtrialremoval(data, events)
                
                predictions = linear.predict(data)
                
                bufhelp.sendevent("classifier.prediction",predictions)  

        elif e.value =="exit":
            run = False
            
        print("Waiting for startPhase.cmd event.")
