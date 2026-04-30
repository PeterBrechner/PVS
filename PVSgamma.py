# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:03:37 2026

@author: peter
"""

import numpy as np
from scipy import io
from scipy.special import gammainc
from scipy.special import gammaln
from scipy.optimize import least_squares
import datetime


def npd(ndarray1, ndarray2):
    return np.divide(ndarray1, ndarray2)

def npm(ndarray1, ndarray2):
    return np.multiply(ndarray1, ndarray2)

def npp(ndarray1, ndarray2):
    return np.power(ndarray1, ndarray2)

def calc_sqE_one_OAP(dD,sd,count):
    """
    Calculate squared bin error in m^-6 for one instrument for fitting class.

    Args:
        dD: dD, in m.
        sd: N(D), in m^-4.
        count: Count in each bin.
    
    Returns:
        sqE for one instrument for fitting class.
    """
    return npd(npp(npm(sd,dD),2),1e-9+count)


class fitting():
    def __init__(self,D,dD,sd,sqE,idx,x,iters):
        """
        Compute gamma fit parameters for PVSs.
        
        Parameters
        ----------
        D: array
            Bin midpoints in m.
        dD: array
            Bin widths in m.
        sd: 2d array
            N(D) in m^-4.
        sqE: 2d array
            Squared bin error in m^-6. The square of the error in N(D)*dD.
        idx: array
            Indices of times to fit. Will fit sd[idx,:] with squared bin error sqE[idx,:].
        x: float
            Exponent x used in m=a*D**x.
        iters: int
            Number of iterations for log_inc in IGF.
        """
        self.D = D
        self.dD = dD
        self.sd = sd[idx,:]
        self.sqE = sqE[idx,:]
        self.idx = idx
        self.x = x
        self.iters = iters
        self.len = len(idx)
        self.M = self.inc_moments()
    
    def get_params(self,file):
        """
        Runs the procedure for calculating needed parameters for PVSs.
        Saves needed parameters to output MAT file.

        Args:
            file: name of output MAT file.
        """
        err = self.sigmas()
        dmin = self.D[0] - 0.5 * self.dD[0]
        dmax = self.D[-1] + 0.5 * self.dD[-1]
        n0, mu, la = self.closed_inc_fit(np.sqrt(err[:,0]), dmin, dmax)
        Mcomp = self.moments(n0, mu, la)
        io.savemat(file+".mat",{"n0": n0, "mu": mu, "la": la, "M": Mcomp, "err": err, "x": self.x})
    
    def calc_mu(self, mu, obs, erry, log_inc):
        """
        Calculates the error vector for mu optimization.

        Args:
            mu: Current estimate for mu.
            obs: Observed moments.
            erry: Squared error for y coordinate.
            log_inc: Log of incomplete gamma function values.

        Returns:
            error_vector: The calculated error.
        """
        mu = mu[0]
        log_gam = np.zeros(3)
        for i in range(3):
            log_gam[i] = log_inc[i] + gammaln(1 + mu + i*self.x)  # use gammaln to avoid overflow
        fmo = (np.log((log_gam[2] - log_gam[1]) + (log_gam[0] - log_gam[1])) 
               - np.log(np.log(obs[2] / obs[1]) + np.log(obs[0] / obs[1])))
        error_vector = fmo / erry
        return error_vector
    
    def closed_inc_fit(self, erry, dmin, dmax):
        """
        Performs closed-form IGF fitting.

        Args:
            erry: Squared errors for y coordinate.
            dmin: Minimum D.
            dmax: Maximum D.

        Returns:
            n0, mu, la: Fitted parameters.
        """
        sz_m = self.M.shape
        mu = np.zeros(sz_m[0])
        la = np.zeros(sz_m[0])
        n0 = np.zeros(sz_m[0])
        starting = np.zeros(sz_m[0])
        upper = np.zeros(sz_m[0])
        for j in range(sz_m[0]):
            upper[j] = 50  # Prevent overflow

        lower = -1
        log_inc = np.zeros((sz_m[0], 3))

        for k in range(self.iters):
            for j in range(sz_m[0]):
                if (j + 1) % 1000 == 1:
                    print(f"Iteration {k+1} is {round(100*(j)/sz_m[0])}% complete: {datetime.datetime.now()}")
                try:
                    result = least_squares(self.calc_mu, starting[j], bounds=(lower, upper[j]),
                                           ftol=1e-4, xtol=1e-4, max_nfev=60, 
                                           args=(self.M[j,:], erry[j], log_inc[j, :]))
                except:
                    print(f"Bad N(D): {self.sd[j,:]}")
                    print(f"Bad index: {self.idx[j]}")
                    raise
                mu[j] = result.x[0]
                #Use m^-1 for lambda for correct loginc with D in m
                la[j] = np.exp(((np.log(self.M[j, 0]) - log_inc[j, 0] - gammaln(1 + mu[j])) -
                                (np.log(self.M[j, 2]) - log_inc[j, 2] - gammaln(1 + mu[j] + 2*self.x))) / (2*self.x))
                for i in range(3):
                    # Ensure indices for gammainc are valid
                    if (la[j] * dmax < 1000): # Heuristic to avoid overflow in gammainc
                        log_inc[j, i] = np.log(gammainc(1 + mu[j] + i*self.x, la[j] * dmax) -
                                               gammainc(1 + mu[j] + i*self.x, la[j] * dmin))
                    else:
                        log_inc[j, i] = np.log(1 - gammainc(1 + mu[j] + i*self.x, la[j] * dmin)) # Indicate potential overflow

                starting[j] = -0.5 + 0.5 * mu[j]
                upper[j] = mu[j]

        for j in range(sz_m[0]):
            if (j + 1) % 1000 == 1:
                print(f"Final iteration is {round(100*(j)/sz_m[0])}% complete: {datetime.datetime.now()}")

            result = least_squares(self.calc_mu, starting[j], bounds=(lower, upper[j]),
                                   ftol=1e-4, xtol=1e-4, max_nfev=60, 
                                   args=(self.M[j,:], erry[j], log_inc[j, :]))
            mu[j] = result.x[0]
            #Use cm^-1 for lambda to support cm^-mu for N0
            la[j] = 1e-2*np.exp(((np.log(self.M[j, 0]) - log_inc[j, 0] - gammaln(1 + mu[j])) - 
                                 (np.log(self.M[j, 2]) - log_inc[j, 2] - gammaln(1 + mu[j] + 2*self.x))) / (2*self.x))
            n0[j] = self.M[j, 1] * 1e2**(self.x - 3) * (la[j]**(1 + mu[j] + self.x)) / np.exp(log_inc[j, 1] + gammaln(1 + mu[j] + self.x))

        return n0, mu, la
    
    def inc_moments(self):
        """
        Calculates incomplete moments in SI units from observed PSDs.

        Returns:
            M: The calculated moments.
        """
        M = np.zeros((self.len,3))
        for i in range(self.len):
            for j in range(3):
                M[i,j] = np.sum(npm(npm(self.sd[i,:],self.dD),npp(self.D,j*self.x)))
        return M
    
    def moments(self, n0, mu, la):
        """
        Calculates complete moments in powers of cm from fitted parameters (N0, mu, lambda).

        Args:
            n0, mu, la: Fitted parameters.

        Returns:
            M: The calculated moments.
        """
        log_m = np.zeros((self.len,3))
        for i in range(3):
            log_m[:,i] = np.log(n0) + gammaln(1 + i*self.x + mu) - (1 + i*self.x + mu) * np.log(la)
        M = np.exp(log_m)
        return M
    
    def sigmas(self):
        """
        Calculates squared errors for y, z, x coordinates.

        Returns:
            err : Squared errors for y, z, x coordinates.
        """
        err = np.zeros((self.len,3))
        for i in range(len(self.D)):
            sqEd = (1/12)*npp(npd(self.dD[i],self.D[i]),2)*self.sqE[:,i] #Accounts for errors in D from coarse bin resolution
            err[:,0] += npm(self.sqE[:,i],npp(npd(1,self.M[:,0])+npd(self.D[i]**(2*self.x),self.M[:,2])-2*npd(self.D[i]**self.x,self.M[:,1]),2))
            err[:,0] += npm(sqEd,npp(npd(2*self.x*self.D[i]**(2*self.x),self.M[:,2])-2*npd(self.x*self.D[i]**self.x,self.M[:,1]),2))
            err[:,1] += (npm(self.sqE[:,i],npp(npd(1,self.M[:,0])-npd(self.D[i]**(2*self.x),self.M[:,2]),2)))/(2*self.x)**2
            err[:,1] += (npm(sqEd,npp(npd(2*self.x*self.D[i]**(2*self.x),self.M[:,2]),2)))/(2*self.x)**2
            err[:,2] += npm(self.sqE[:,i],npp(npd(self.D[i]**self.x,self.M[:,1]),2))
            err[:,2] += npm(sqEd,npp(npd(self.x*self.D[i]**self.x,self.M[:,1]),2))
        err[:,0] = npd(err[:,0],npp(np.log(self.M[:,0])+np.log(self.M[:,2])-2*np.log(self.M[:,1]),2))
        return err