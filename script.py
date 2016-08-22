from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
from scipy.linalg import inv

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD

    train_data = np.array([]).astype('float').reshape(0, 3)
    for i in range(0, X.shape[0]):
        train_data = np.vstack((train_data, [y[i], X[i][0], X[i][1]]))

    fm = np.array([]).astype('float')
    sm = np.array([]).astype('float')

    for i in range(1, 6):
        a = train_data[train_data[:, 0] == i, :]
        fm = np.append(fm, np.mean(a[:, 1]))
        sm = np.append(sm, np.mean(a[:, 2]))

    means = np.array([]).astype('float').reshape(0, 5)
    means = np.vstack((means, fm))
    means = np.vstack((means, sm))

    XT = np.transpose(X)
    covmat = np.cov(XT)

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    train_data = np.array([]).astype('float').reshape(0, 3)
    for i in range(0, X.shape[0]):
        train_data = np.vstack((train_data, [y[i], X[i][0], X[i][1]]))

    fm = np.array([]).astype('float')
    sm = np.array([]).astype('float')

    covmats = []
    list = np.array([]).astype('float')
    for i in range(1, 6):
        a = train_data[train_data[:, 0] == i, :]
        fm = np.append(fm, np.mean(a[:, 1]))
        sm = np.append(sm, np.mean(a[:, 2]))
        cov = np.cov(np.transpose(a[:, [1, 2]]))
        covmats.append([cov])

    means = np.array([]).astype('float').reshape(0, 5)
    means = np.vstack((means, fm))
    means = np.vstack((means, sm))

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    acc = 0.0
    ci = np.linalg.inv(covmat)

    f = np.array([]).astype('float');
    ypred = np.array([])
    for k in range(0, Xtest.shape[0]):
        a = np.array(()).astype('float')
        for i in range(0, 5):
            f = means[:, i] * np.dot(ci, np.transpose(Xtest[k])) - (
            0.5 * means[:, i] * np.dot(ci, np.transpose(means[:, i])))
            a = np.append(a, np.mean(f))
        ypred = np.append(ypred, np.argmax(a) + 1)

    ypred = ypred.reshape(ypred.shape[0], 1)
    acc = round(np.mean((ypred == ytest)), 2)

    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    f = np.array([]).astype('float');
    ypred = np.array([])
    for k in range(0, Xtest.shape[0]):
        a = np.array(()).astype('float')
        for i in range(0, 5):
            cv = np.asarray(covmats[i][0]).astype('float')
            ci = inv(cv).reshape(2, 2)
            deno = sqrt(np.linalg.det(2 * pi * cv))
            x_min_mean = Xtest[k] - means[:,i]
            temp1 = -0.5 * x_min_mean.transpose()
            exp = np.dot(np.dot(temp1, cv), x_min_mean)
            result = (1/deno) * np.exp(exp)
            a = np.append(a, result)

        ypred = np.append(ypred, np.argmax(a) + 1)

    ypred = ypred.reshape(ypred.shape[0], 1)
    acc = round(np.mean((ypred == ytest)), 2)

    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD
    ai = np.linalg.inv(np.dot(np.transpose(X), X))
    b = np.dot(ai, np.transpose(X))
    w = np.dot(b, y)

    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    d = X.shape[1]
    transpose_x = np.transpose(X)
    identity_matrix = np.identity(d)
    temp1 = inv(np.dot(transpose_x, X) + lambd * identity_matrix)
    temp2 = np.dot(transpose_x, y)
    w = np.dot(temp1, temp2)

    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    # IMPLEMENT THIS METHOD
    wtx = np.dot(Xtest, w)
    rmse = sqrt(np.sum(np.power((ytest - wtx), 2)) / ytest.shape[0])
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect.
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHODw = nnparams
    transpose_w = np.transpose(w)
    transpose_x = np.transpose(X)

    w = np.array(w).reshape(w.size, 1)
    XW = np.dot(X, w)
    ymxw = np.subtract(y, XW)
    error = 0.5 * np.dot(np.transpose(ymxw), ymxw) + 0.5 * lambd * np.dot(np.transpose(w), w)

    xtx = np.dot(np.transpose(X), X)
    xtxw = np.dot(xtx, w)
    xty = np.dot(np.transpose(X), y)
    error_grad = np.subtract(xtxw, xty)
    error_grad = np.add(error_grad, lambd * w)
    error_grad = error_grad.flatten()

    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    # IMPLEMENT THIS METHOD
    Xd = np.array([]).astype('float').reshape(x.size, 0)
    for i in range(0, p+1):
        a = np.power(x, i)
        Xd = np.column_stack((Xd, a))

    return Xd

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
#plt.show()


zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
#plt.show()

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE test without intercept '+str(mle))
print('RMSE test with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)
#plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='BFGS', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)
#plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
#plt.show()

