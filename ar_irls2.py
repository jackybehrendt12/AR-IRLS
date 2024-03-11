from PySide6.QtWidgets import QApplication, QFileDialog
import sys
import mne
import nilearn
from nilearn.glm.first_level import (
    make_first_level_design_matrix 
)
import h5py
from mne.io import read_raw_snirf
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt


# Function to select input file

def select_input_file():
    # Check if QApplication instance exists
    app = QApplication.instance()
    # If not, create one
    if not app:
        app = QApplication(sys.argv)
    # Open file dialog to select fNIRS data file
    file_path, _ = QFileDialog.getOpenFileName(
        None, "Select fNIRS data file", "", "SNIRF files (*.snirf)")
    return file_path

def robustfit(X,y, tune = 4.685 ,const='on',dowarn= True ): 

    if const == 'on':
        doconst = True
    else:   
        doconst = False 
    
    def wfit(y, x, w): 
        n = np.shape(x)[1]
        sw = np.sqrt(w)
        yw = y * sw.T  
        xw = np.multiply(x, sw.T)
        lstsq = np.linalg.lstsq(xw, yw, rcond=None)
        b = lstsq[0]
        r = lstsq[2]    

        return b, r 

    def madsigma(r, p ): 
        rs = np.sort(np.abs(r)) [0]
        s = np.median(rs[np.maximum(1,p)-1:]) / 0.6745  
        return s 

    def wfun(r): 
        w = (np.abs(r)<1) * np.power((1 - np.power(r, 2)), 2)
        return w 


    def statrobustfit(X,y,wfun,tune,wasnan,addconst,dowarn): 
        n = np.shape(X)[0]
        p = np.shape(X)[1] 
        if addconst:
            X = np.concatenate((np.ones((n,1)),X),axis=1) 
            p = p+1
        if n <= p : 
            raise ValueError('stats:statrobustfit:TooFewDataPoints')    
        
        sw = 1 
        Q, R, perm = sp.linalg.qr(X, mode='economic', pivoting=True) 

        b = np.zeros((X.shape[1], y.shape[1]))   
        if np.allclose(R, 0):
            tol = 1
        else:
            tol = abs(R[0, 0]) * max(n, p) * np.finfo(R.dtype).eps  
        
        xrank = np.sum(np.abs(np.diag(R)) > tol)    
        if xrank == p: 
            b[perm, : ] = np.linalg.solve(R, np.dot(Q.T, y))    
        else:
            if dowarn:
                print("Rank-deficient:", xrank) 
            b[perm, : ] = np.concatenate((np.linalg.solve(R[:xrank, :xrank], np.dot(Q[:, :xrank].T, y)), np.zeros((p-xrank, 1 )) )  )
            perm = perm[:xrank]
        b0 = np.zeros(b.shape)  

        E = np.dot(X[:, perm], np.linalg.inv(R[:xrank, :xrank]  ) )
        
        # product with E and E
        h = np.minimum(.9999, np.sum(np.power(E, 2),1))
        
        adjfactor = 1/ np.sqrt(1-h)    

        dfe = n-xrank
        ols_s = np.linalg.norm((y- np.dot(X, b))/sw )/ np.sqrt(dfe)
        
        
        tiny_s = 1e-6* np.std(y)
        if tiny_s == 0 : 
            tiny_s = 1 
        
        D = np.sqrt(np.finfo(X.dtype).eps )
    
        
        iter = 0 
        iterlim =  50 
        wxrank = xrank 

        while iter == 0 and np.any(np.abs(b-b0) > D* np.maximum(np.abs(b), np.abs(b0))):
            iter = iter + 1 
            if iter > iterlim:
                raise ValueError('stats:statrobustfit:IterationLimit')
            
            r = y - np.dot(X, b)   
            radj = np.divide(np.multiply(r.T, adjfactor), sw )
            s = madsigma(radj, wxrank)
            w = wfun(radj/(np.maximum(s,tiny_s)*tune))  
            b0 = b 
            b[perm], wxrank = wfit(y, X[:, perm], w)
        
        return b 
        
    
    b = statrobustfit(X,y,wfun,tune, [], doconst,dowarn)

    return b



def standard_least_squares(d, X):

    B = np.zeros((np.shape(X)[1], np.shape(d)[1]))
    nChan = np.shape(d)[1]

    for i in range(0, nChan):
        y = d[:, i ]
        # initial fit
        lstValid = ~np.isnan(y)
        b = np.dot(np.linalg.pinv(X[lstValid, :]), y[lstValid])
        error = np.linalg.norm(y[lstValid] - np.dot(X[lstValid, : ], b) ) 
        B[:,i] =  b 

    return B  



def lagmatrix(y, lags):
    n = np.shape(y)[0]  
    N = np.max(lags)
    y2 = y.flatten()
    ylag = sp.linalg.convolution_matrix(y2, N+1, 'full')
    ylag = ylag[:n, lags.flatten()]
    return ylag 

def infocrit( LogL , num_obs , num_param , criterion = 'BIC' ): 
    #Calculate information criterion from Log-Likelihood (BIC, AIC, AICc, CAIC, MAX)
    #crit = infocrit( LogL , num_obs , num_param , criterion )
    num_param = np.reshape(num_param, (np.shape(num_param)[0], 1))  
    
    criterion = criterion.upper()
    if criterion == 'BIC':
        crit = -2*LogL + num_param*np.log(num_obs)
    if criterion == 'AIC':
        crit = -2*LogL + 2*num_param
    if criterion == 'AICC':
        crit = -2*LogL + 2*num_param + 2*num_param*(num_param+1)/(num_obs-num_param-1)  
        index = (num_obs-num_param-1) <= 0
        crit[index.flatten()] = np.nan   
    if criterion == 'CAIC':
        crit = -2*LogL + num_param*(np.log(num_obs)+1)

    if criterion not in ['BIC', 'AIC', 'AICC', 'CAIC']: 
        raise OSError('Unknown model selection criterion: ' + criterion) # which kind of error here? 

    return crit 




def stepwise(X, y, criterion = 'BIC'):
    # qr factorization will speed up stepwise regression significantly

    Q, R = np.linalg.qr(X)
    invR = np.linalg.pinv(R)  
    n = np.shape(y)[0]  
    LL = np.full((np.shape(X)[1], 1), np.nan )  

    for i in range(0, np.shape(LL)[0]): 
        # get residual for each fit

        b = invR[0:i+1, 0:i+1].dot(np.transpose(Q[:,:i+1]).dot(y)) 
        r = y - X[:, :i+1].dot(b) 
        # calculate log-likelihood
        LL[i] = -0.5*n*np.log(np.sum(r**2)/n) - 0.5*n*np.log(2*np.pi) - 0.5*n
    
    # Calculate information criterion

    crit = infocrit(LL, n, np.arange(1, np.shape(LL)[0]+1).T, criterion)
    
    #optimal model order
    lst = np.where(~np.isnan(crit))[0]
    N = np.argmin(crit[lst])
    N = lst[N]   

    b = invR[0:N+1, 0:N+1].dot(np.transpose(Q[:,:N+1]).dot(y))  
    r = y - X[:, :N+1].dot(b)  

    return b , r, crit 







def ar_fit(y, Pmax, nosearch = False):
    n = np.shape(y)[0]
    Pmax = np.maximum(np.minimum(np.min(Pmax), n-1) ,1 )

    Xf = lagmatrix(y, np.arange(1, Pmax+1))
    Xb = lagmatrix(np.flipud(y), np.arange(1, Pmax+1)) 
    X = np.concatenate((np.ones((2*n, 1)), np.concatenate((Xf, Xb), axis = 0)), axis = 1)
    yy = np.concatenate((y, np.flipud(y)), axis = 0)     

    lstValid = np.logical_and(~np.isnan(yy), np.reshape(~np.isnan(np.sum(X,1)), np.shape(yy)))
  
    if nosearch: 
        Q, R = np.linalg.qr(X[lstValid.flatten(), :])  
        invR = np.linalg.pinv(R)
        coef = invR.dot(Q.T.dot(yy[lstValid.flatten()])) 
        res = yy[lstValid.flatten()] - X[lstValid.flatten(), : ].dot(coef) 

    else: 
        coef, res, _ = stepwise(X[lstValid.flatten(), :], yy[lstValid.flatten()])
        

    res = res[:n ] 
    yhat = y -res 

    return coef, res, yhat



def myFilter( f, y ): 
    y1 = y[0, :]
    y = y-y1
    out = sp.signal.lfilter (f, 1, y, axis = 0) 
    out = out +  np.sum(f)* y1
    return out

def ar_irls(d,X,Pmax,tune = 4.685, nosearch=False ,useGPU=False, singlePrecision=False): 
    d = np.array(d)
    X = np.array(X)
    B_final = np.zeros((np.shape(X)[1], np.shape(d)[1])) 

    for i in range(0, nChan): 
        y = d[:, i ]
        # initial fit
        lstValid = ~np.isnan(y)
            
        B = np.dot(np.linalg.pinv(X[lstValid, :]), y[lstValid])

        
        B0 = 1e6* np.ones(np.shape(B)) 

        # compute sigma
        sigma = np.sqrt(np.var(y[lstValid] - np.dot(X[lstValid, :], B)))

        # iterative re-weighted least squares
        iter1 = 0
        maxiter = 10 
            
        # while our coefficients are changing greater than some threshold
        # and its less than the max number of iterations

        while np.linalg.norm(B-B0)/np.linalg.norm(B0) > 1e-2 and  iter1 < maxiter: #make sure this runs more often .....

            B0 = B                  # store the last fit
            y = np.reshape(y, (np.shape(y)[0],))
            res = y - np.dot(X, B)  #get the residual 
            
    

            if  len(np.shape(Pmax)) > 1: 
                p = Pmax[i]
            else: 
                p = Pmax

            res = np.array(res) 
            res = np.reshape(res, (np.shape(res)[0], 1))    
            a, _ , _  = ar_fit( res, p, nosearch) 
        


            # create a whitening filter from the coefficients
            f = np.concatenate((np.array([1]), -a[1:].flatten()))
            # filter the design matrix
            Xf = myFilter(f,X)
        

            lstInValid = np.isnan(y)
            lstValid = ~np.isnan(y)
                
            if np.count_nonzero(lstInValid) > 0: 
                yy = y 
                integral = sp.interpolate.interp1d(np.where(lstValid > 0 )[0], y[lstValid], kind = 3 )
                # warning: what about boundary points ?? 
                yy[lstInValid.flatten()] = integral(np.where(lstInValid > 0 )[0])
                yy = np.reshape(yy, (np.shape(yy)[0], 1))
                yf = myFilter(f, yy)
            else: 
                y = np.reshape(y, (np.shape(y)[0], 1))
                yf = myFilter(f,y )
                
         
            # perform IRLS
            B = robustfit(Xf[lstValid,:],yf[lstValid],  tune, 'off')
            B = np.reshape(B, (np.shape(B)[0], ))  
            iter1 = iter1 + 1 

         
        
        B_final[:,i] = B
    
    return B_final


# Select input files
"""
input_file = select_input_file()

# Read SNIRF data from the selected files
raw = read_raw_snirf(input_file)

# Get data from raw object

data = raw.get_data()
data = data[0:9, 0:600]    #test  


np.save('data', data)

data = np.load('data.npy')    #test
data = np.transpose( np.reshape(data, (9, 600 )))  
ntime = np.shape(data)[0]
nChan = np.shape(data)[1]  
d = data 

# build design matrix   

frames = np.arange(0, ntime, 1)   
X = make_first_level_design_matrix(frames ,drift_model='polynomial', hrf_model='glover')
X = np.array(X)  
print('Design Matrix Shape: ', np.shape(X) )    


B_ls = standard_least_squares(d, X)  
prediction_ls = np.dot(X, B_ls)

print('Standard Least Squares Error: ', int(np.linalg.norm(d - prediction_ls)) )

Pmax = 4 * ntime    

B_ar = ar_irls(d, X, Pmax )
prediction_ar = np.dot(X, B_ar) 
print('AR IRLS Error:    ', int(np.linalg.norm(d - prediction_ar)) )


import matlab.engine
eng = matlab.engine.start_matlab()
d2 =matlab.double(d.tolist())
X2 = matlab.double(X.tolist())
result = eng.ar_irls(d2, X2, Pmax )
B_m = eng.getfield(result, 'beta')
prediction_m = np.dot(X, B_m) 
print('Matlab AR IRLS Error:    ', int(np.linalg.norm(d - prediction_m)) )
eng.quit() 


k = 0 
fig, ax = plt.subplots(3, 3)
for i in range(3):  
    for j in range(3): 
        l1 = ax[i, j].plot(d[:, k ], 'r', prediction_ls[:,k] , 'b' , prediction_ar[:,k], 'g'  , prediction_m[:,k], 'k' ) 
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        k = k+1 


labels = ["Data", "Standard Least Squares", "AR IRLS", "Matlab AR IRLS"] 
fig.legend( labels=labels, loc="upper right")

plt.show()  """
