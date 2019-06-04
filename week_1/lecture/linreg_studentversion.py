import numpy as np, os, sys

import matplotlib as mpl #patch-wise similarities, droi images

def datagen2d(w1,w2,eps,num):
  """
  num: number of samples
  """
  X = np.random.normal(0, 1, size=(num,2))
  weights = np.array([w1, w2]).reshape(-1,1)
  n = np.random.normal(0, eps)

  y = X@weights+n
  return X, y

def rndsplit(x,y,numtr):

  inds=np.arange(y.size)
  np.random.shuffle(inds)

  xtr=x[inds[0:numtr],:]
  ytr=y[inds[0:numtr]]

  xv=x[inds[numtr:],:]
  yv=y[inds[numtr:]]

  return xtr,ytr,xv,yv

def gendata(numtotal):
  
  w1=0.5
  w2=-2
  xtr,ytr = datagen2d(w1=w1,w2=w2,eps=0.8,num= int( numtotal) ) #numtotal
  xv,yv = datagen2d(w1=w1,w2=w2,eps=0.8,num= 3000 ) #fix 3k test samples
  #xtr,ytr,xv,yv = rndsplit(x,y,numtr=int( 0.7*numtotal))
  return xtr,ytr,xv,yv,w1,w2

def gendata2(numtotal):
  
  w1=0.5
  w2=-2
  xtr,ytr = datagen2d(w1=w1,w2=w2,eps=0.8,num= int( 0.7*numtotal) ) #70% of numtotal
  xv,yv = datagen2d(w1=w1,w2=w2,eps=0.8,num= int( 0.3*numtotal) ) #30 % of numtotal
  #xtr,ytr,xv,yv = rndsplit(x,y,numtr=int( 0.7*numtotal))
  return xtr,ytr,xv,yv,w1,w2

def linreg_train(xtr,ytr,C):
  N, d  = xtr.shape
  # your implementation here
  w = (xtr.T@xtr+C*np.eye(d)) @ (xtr.T@ytr)
  return w

def linreg_apply(xv,w):
  return np.dot(xv,w)

def mse(ypred,ytrue):
  e=np.mean( (ypred-ytrue)**2 )
  return e


def run1(xtr,ytr,xv,yv,w1,w2,C):

  w=linreg_train(xtr,ytr,C=C) # 0.1

  wtrue=np.asarray([w1,w2])

  print('w',w, 'true w', [w1,w2], 'diff', np.dot((w-wtrue).T,w-wtrue))

  ypred=linreg_apply(xv,w)
  e=mse(ypred,yv)

  print('mse',e)

if __name__=='__main__':

  xtr,ytr,xv,yv,w1,w2=gendata(50)
  run1(xtr,ytr,xv,yv,w1,w2,1e-3)

  xtr,ytr,xv,yv,w1,w2=gendata(1000)  
  run1(xtr,ytr,xv,yv,w1,w2,1e-3)
