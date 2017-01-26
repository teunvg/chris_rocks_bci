def calcStayProb(mEvents):
    return mEvents / (mEvents+1)

mLRP = 3.
mERD = 6.
mERS = 4.
mIDLE = 6.
nEvents = mLRP+mERD+mERS
pIDLE = mIDLE / (nEvents + mIDLE)
restProbs = 1 - pIDLE
pLRP = restProbs * mLRP/nEvents/3
pERD = restProbs * mERD/nEvents/3
pERS = restProbs * mERS/nEvents/3
pIDLE = 1 - (3 * pLRP + 3 * pERD + 3 * pERS)

HMM.startprob_ = np.array([pIDLE] + [pLRP] * 3 + [pERD] * 3 + [pERS] * 3)

psIDLE = (1/p)/((1/p)+1)
psLRP = calcStayProb(mLRP)
psERD = calcStayProb(mERD)
psERS = calcStayProb(pERS)

transitionMatrix = np.matrix([
    [.7, .1, .1, .1, 0, 0, 0, 0, 0, 0],
    [0, psLRP, 0, 0, 1-psLRP, 0, 0, 0, 0, 0],
    [0, 0, psLRP, 0, 0, 1-psLRP, 0, 0, 0, 0],
    [0, 0, 0, psLRP, 0, 0, 1-psLRP, 0, 0, 0],
    [0, 0, 0, 0, psERD, 0, 0, 1-psERD, 0, 0],
    [0, 0, 0, 0, 0, psERD, 0, 0, 1-psERD, 0],
    [0, 0, 0, 0, 0, 0, psERD, 0, 0, 1-psERD],
    [1-psERS, 0, 0, 0, 0, 0, 0, psERS, 0, 0],
    [1-psERS, 0, 0, 0, 0, 0, 0, 0, psERS, 0],
    [1-psERS, 0, 0, 0, 0, 0, 0, 0, 0, psERS]
    ])