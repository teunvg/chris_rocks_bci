import sklearn.linear_model
import skwrap

def fit(data, events, mapping=dict()):
    classifier = sklearn.linear_model.LinearRegression()
    params = {"fit_intercept" : [True, False], "normalize" : [True, False]}             
    skwrap.fit(data, events, classifier, mapping, params, reducer="mean")
    return classifier
    
def predict(classifier, data):
    return skwrap.predict(data, classifier, reducerdata="mean")

def probabilities(classifier, data):
    return skwrap.predict_proba(data, classifier, reducerdata="mean")