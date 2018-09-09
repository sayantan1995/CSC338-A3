import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.linalg as nla
import pickle

# Question 7


def train(data):

    d = len(data)
    Sigma = []
    mu = []
    var = np.zeros(d)
    num = np.zeros(d)

    for i in range(0, d):
        [m,S] = fitNormal(data[i])
        mu.append(m)
        Sigma.append(S)
        var[i] = np.mean(np.diag(S))
        num[i] = np.shape(data[i])[0]

    meanVar = np.sum(var * num)/np.sum(num)
    return (mu, Sigma, meanVar)


# Question 8

# part a

def flatten(data):

    d = len(data)
    X = np.vstack(data)
    M = np.shape(X)[0]
    Y = np.zeros(M, dtype = 'int')
    m1 = 0
    m2 = 0

    for i in range(0, d):
        m = np.shape(data[i])[0]
        m2 += m
        Y[m1:m2] = i
        m1 += m

    return(X, Y)

#part b

def combine(Sigma, var, beta):
    d = len(Sigma)
    N = np.shape(Sigma[0])[0]
    Simple = (1 - beta) * var * np.eye(N)
    Combined = []

    for i in range (0, d):
        C = beta * Sigma[i] + Simple
        Combined.append(C)

    return(Combined)

# part c

def logMVNchol(X, mu, Sigma):

    [M,N] = X.shape
    logP = np.zeros(M)
    L = la.cholesky(Sigma, lower = True)
    logDet = nla.slogdet(Sigma)[1]
    logAlpha = np.log(2 * np.pi) * N/2. + logDet

    for m in range(0, M):
        x = X[m,:] - mu
        y = la.solve_triangular(L, x, lower = True, check_finite = False)
        logP [m] = -np.dot(y, y)/2. - logAlpha

    return(logP)

# part d

def predict(X, mu, Sigma):

    M = np.shape(X)[0]
    d = len(mu)
    logP = np.zeros((M,d))

    for i in range(0, d):
        logP[:,i] = logMVNchol(X, mu[i], Sigma[i])

    return(logP)

# Question 9

# part a

def evaluate(logP, X, Y):

    Yhat = np.argmax(logP, axis = 1)
    right = (Y == Yhat)
    M = np.shape(X)[0]
    accuracy = 100 * np.sum(right)/float(M)
    print(accuracy)
    N = 36
    Correct = X[right,:]
    rnd.shuffle(Correct)
    showImages(N, Correct, 'Some correctly classified images')
    Errors = X[~right,:]
    rnd.shuffle(Errors)
    showImages(N, Errors, 'Some misclassified images')

# part c

def ocr():
    with open('mnist.pickle', 'rb') as f:
        data = pickle.load(f)

    (mu, Sigma, var) = train(data['training'])
    print('training done')
    (X, Y) = flatten(data['testing'])
    beta = 0.25
    SigmaNew = combine(Sigma, var, beta)
    logP = predict(X, mu, SigmaNew)
    print('prediction done')
    evaluate(logP, X, Y)
    print('evaluation done')

# From A1 and A2
# 7(b)

def fitNormal(X):
    (M, N) = X.shape
    mu = X.sum(axis = 0)/float(M)
    Xc = X - mu
    Sigma = np.dot(Xc.T, Xc)/float(M)
    return (mu, Sigma)

def showImages(N, data, title):
    fig = plt.figure()
    fig.suptitle(title)
    M = int(np.ceil(np.sqrt(N)))
    m = int(np.sqrt(np.size(data[0])))

    for i in range(0, N):
        x = data[i]
        y = np.reshape(x, (m, m))
        plt.subplot(M, M, i + 1)
        plt.axis('off')
        plt.imshow(y, cmap = 'Greys', interpolation = 'nearest')



