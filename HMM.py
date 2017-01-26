class HMM:
    A = np.matrix([[]])
    B = np.matrix([[]])
    
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def alpha(self, X, j, t):
        if t == 0:
            return 0
        else:
            return sum([self.alpha(X, i, t - 1) * self.A[i, j] * self.B[j, X[t]] for i in range(self.A.shape[0])])

    def forward(self, X, T):
        return sum([self.alpha(X, i, T) for i in range(self.A.shape[0])])

    