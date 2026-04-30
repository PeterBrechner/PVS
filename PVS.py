# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:09:11 2026

@author: peter
"""

import numpy as np
from numpy import linalg
from scipy import stats
from scipy.special import gammaln
from scipy.stats.distributions import chi2
from scipy.stats import norm
from scipy.optimize import least_squares
import pylab
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision = 3)


def npd(ndarray1, ndarray2):
    return np.divide(ndarray1, ndarray2)

def npm(ndarray1, ndarray2):
    return np.multiply(ndarray1, ndarray2)

def npp(ndarray1, ndarray2):
    return np.power(ndarray1, ndarray2)


class PVS():
    def __init__(self,M,err,x=3,chisq=4,norm=False,res=241):
        """
        Create a PVS object.
        
        Parameters
        ----------
        M: 2d array
            M from PVSgamma output MAT file.
        err: 2d array
            err from PVSgamma output MAT file.
        x: float
            x from PVSgamma output MAT file. The default is 3.
        chisq: float, optional
            chi squared for PVS surface. The default is 4.
        norm: Boolean, optional
            Normalized PVSs if True, real PVSs if False. The default is False.
        res: odd int, optional
            Number of distinct values of mu used for plotting. The default is 241.
        """
        self.M = M
        self.err = err
        self.x = x
        self.chisq = chisq
        self.norm = norm
        self.res = res

        y_coord = -np.log(np.log(self.M[:,0])+np.log(self.M[:,2])-2*np.log(self.M[:,1]))
        z_coord = (np.log(self.M[:,0])-np.log(self.M[:,2]))/(2*self.x)
        x_coord = np.log(self.M[:,1])
        self.ym = np.mean(y_coord)
        self.ys = np.sqrt(np.var(y_coord)+np.mean(self.err[:,0]))
        self.ysm = np.sqrt(np.mean(self.err[:,0]))
        self.zm = np.mean(z_coord)
        self.zs = np.sqrt(np.var(z_coord)+np.mean(self.err[:,1]))
        self.xm = np.mean(x_coord)
        self.xs = np.sqrt(np.var(x_coord)+np.mean(self.err[:,2]))
        Y = [y_coord-self.ym]
        if self.norm:
            self.a = [0]
            self.ra = 0
        else:
            self.a = linalg.lstsq(np.transpose(Y),np.transpose(z_coord),rcond=None)[0]
            self.ra = stats.linregress(Y,z_coord,axis=None)[2]
        self.zs = self.zs*np.sqrt(1-self.ra**2)
        Z = [z_coord-(self.zm+(y_coord-self.ym)*self.a[0]),y_coord-self.ym]
        if self.norm:
            self.b = [0,0]
            self.rb = 0
        else:
            self.b = linalg.lstsq(np.transpose(Z),np.transpose(x_coord),rcond=None)[0]
            fitb = self.b[0]*Z[0]+self.b[1]*Z[1] #b0*la+b1*mu
            self.rb = stats.linregress(fitb,x_coord,axis=None)[2]
        self.xs = self.xs*np.sqrt(1-self.rb**2)
        #to save time, calculate mu exactly once per PVS object
        psi = np.linspace(0,np.pi,self.res)
        self.mu_ML = self.find_mu(self.ym)
        mu = np.zeros(len(psi))
        for i in range(len(psi)):
            mu[i] = self.find_mu(self.ym-np.cos(psi[i])*np.sqrt(self.chisq)*self.ys)
        self.mu = np.transpose(mu)
    
    def chi2x(self,y,z,x):
        """
        x term in chi squared. Represents likelihood of N0 value.

        Args:
            y: Known value for y coordinate.
            z: Known value for z coordinate.
            x: Known value for x coordinate.

        Returns:
            x term in chi squared.
        """
        return (x-(self.xm+self.b[0]*self.zs*self.chi2z(y,z)+self.b[1]*self.ys*self.chi2y(y)))/self.xs
    
    def chi2y(self,y):
        """
        y term in chi squared. Represents likelihood of mu value.

        Args:
            y: Known value for y coordinate.

        Returns:
            y term in chi squared.
        """
        return (y-self.ym)/self.ys
    
    def chi2z(self,y,z):
        """
        z term in chi squared. Represents likelihood of lambda value.

        Args:
            y: Known value for y coordinate.
            z: Known value for z coordinate.

        Returns:
            z term in chi squared.
        """
        return (z-(self.zm+self.a[0]*self.ys*self.chi2y(y)))/self.zs
    
    def find_mu(self,y):
        """
        Calculates mu for PVS.

        Args:
            y: Known value for y coordinate.

        Returns:
            mu: Shape parameter.
        """
        starting = 0
        upper = 50  # Prevent overflow
        lower = -1
        try:
            result = least_squares(self.solve_mu, starting, bounds=(lower, upper), 
                                   ftol=1e-4, xtol=1e-4, max_nfev=60, args=(y,))
        except:
            raise
        mu = result.x[0]
        return mu
    
    def Plot(self,color,dash=False,area=False):
        """
        Plot the perimeter of a mu-lambda PVS cross-section and return gamma fit parameters for the cross-section.
        
        Args:
            color: color
                Color for plot.
            dash: Boolean, optional
                True for a dashed perimeter and False for a solid perimeter. The default is False.
            area: Boolean, optional
                If True, returns gamma fit parameters for the area of the cross-section. The default is False.
        
        Returns:
            mu_ML: float
                mu for most likely gamma fit parameters.
            la_ML: float
                lambda for most likely gamma fit parameters.
            mu: array
                mu for perimeter of mu-lambda cross-section.
            la_l: array
                lambda for lower half of perimeter of mu-lambda cross-section.
            la_u: array
                lambda for upper half of perimeter of mu-lambda cross-section.
        
        If area, also returns:
            logN0_area: 2d array
                log10(N0) for area of mu-lambda cross-section.
            mu_area: 2d array
                mu for area of mu-lambda cross-section.
            la_area: 2d array
                lambda for area of mu-lambda cross-section.
        """
        psi = np.linspace(0,np.pi,self.res)
        yp = -np.log(gammaln(1+2*self.x+self.mu)+gammaln(1+self.mu)-2*gammaln(1+self.x+self.mu))
        sig2l = np.sqrt(self.chisq)*np.sin(np.transpose(psi))
        la = 0.1*np.exp(self.zm+(yp-self.ym)*self.a[0]+(gammaln(1+2*self.x+self.mu)-gammaln(1+self.mu))/(2*self.x))
        la_u = npm(la,np.exp(self.zs*sig2l))
        la_l = npm(la,np.exp(-self.zs*sig2l))
        la_ML = 0.1*np.exp(self.zm+(gammaln(1+2*self.x+self.mu_ML)-gammaln(1+self.mu_ML))/(2*self.x))
        logN0 = 8+((self.xm+(yp-self.ym)*self.b[1])-gammaln(1+self.x+self.mu)+npm((1+self.x+self.mu),np.log(10*la)))/np.log(10)
        if dash:
            pylab.plot(self.mu,la_l,color=color,lw=2,ls='--')
            pylab.plot(self.mu,la_u,color=color,lw=2,ls='--')
        else:
            pylab.plot(self.mu,la_l,color=color,lw=2)
            pylab.plot(self.mu,la_u,color=color,lw=2)
        if area:
            phi = np.linspace(0,2*np.pi,self.res)
            logN0_area = np.zeros((len(self.mu),len(phi)))
            la_area = np.zeros((len(self.mu),len(phi)))
            mu_area = np.zeros((len(self.mu),len(phi)))
            for i in range(len(phi)):
                sig2n = self.zs*npm(sig2l,np.sin(phi[i]))
                logN0_area[:,i] = logN0+npm(1+self.x+self.mu+self.b[0],sig2n)/np.log(10)
                la_area[:,i] = npm(la,np.exp(sig2n))
                mu_area[:,i] = self.mu
            return (self.mu_ML,la_ML,self.mu,la_l,la_u,logN0_area,mu_area,la_area)
        else:
            return (self.mu_ML,la_ML,self.mu,la_l,la_u)

    def PVS(self,jumpup=True):
        """
        Plot the perimeter of a mu-lambda PVS cross-section.
        
        Args:
            jumpup: Boolean, optional
                If True, projection of N0 onto mu-lambda cross-section jumps up 
                as lambda increases across its most likely value.
                If False, projection of N0 onto mu-lambda cross-section jumps down 
                as lambda increases across its most likely value.
        
        Returns:
            logN0_ML: float
                log10(N0) for most likely gamma fit parameters.
            mu_ML: float
                mu for most likely gamma fit parameters.
            la_ML: float
                lambda for most likely gamma fit parameters.
            logN0_surf: 2d array
                log10(N0) for surface of 3D PVS.
            mu_surf: 2d array
                mu for surface of 3D PVS.
            la_surf: 2d array
                lambda for surface of 3D PVS.
        """
        psi = np.linspace(0,np.pi,self.res)
        yp = -np.log(gammaln(1+2*self.x+self.mu)+gammaln(1+self.mu)-2*gammaln(1+self.x+self.mu))
        sig2l = np.sqrt(self.chisq)*np.sin(np.transpose(psi))
        la = 0.1*np.exp(self.zm+(yp-self.ym)*self.a[0]+(gammaln(1+2*self.x+self.mu)-gammaln(1+self.mu))/(2*self.x))
        la_ML = 0.1*np.exp(self.zm+(gammaln(1+2*self.x+self.mu_ML)-gammaln(1+self.mu_ML))/(2*self.x))
        logN0 = 8+((self.xm+(yp-self.ym)*self.b[1])-gammaln(1+self.x+self.mu)+npm((1+self.x+self.mu),np.log(10*la)))/np.log(10)
        logN0_ML = 8+(self.xm-gammaln(1+self.x+self.mu_ML)+npm((1+self.x+self.mu_ML),np.log(10*la_ML)))/np.log(10)
        theta = np.arctan(npm((1+self.x+self.mu+self.b[0]),self.zs/self.xs))
        phi = np.linspace(0,2*np.pi,self.res)
        logN0_surf = np.zeros((len(self.mu),len(phi)))
        la_surf = np.zeros((len(self.mu),len(phi)))
        mu_surf = np.zeros((len(self.mu),len(phi)))
        if jumpup:
            for i in range(len(phi)):
                logN0_surf[:,i] = logN0+(self.xs*npm(npd(sig2l,np.cos(theta)),np.cos(-phi[i]-theta)))/np.log(10)
                la_surf[:,i] = npm(la,np.exp(self.zs*npm(sig2l,np.sin(-phi[i]))))
                mu_surf[:,i] = self.mu
        else:
            for i in range(len(phi)):
                logN0_surf[:,i] = logN0+(self.xs*npm(npd(sig2l,np.cos(theta)),np.cos(phi[i]-theta)))/np.log(10)
                la_surf[:,i] = npm(la,np.exp(self.zs*npm(sig2l,np.sin(phi[i]))))
                mu_surf[:,i] = self.mu
        return (logN0_ML,self.mu_ML,la_ML,logN0_surf,mu_surf,la_surf)
    
    def solve_mu(self, mu, y):
        """
        Calculates the error vector for mu optimization.

        Args:
            mu: Current estimate for mu.
            y: Known value for y coordinate.

        Returns:
            error_vector: The calculated error.
        """
        mu = mu[0]
        log_gam = np.zeros(3)
        for i in range(3):
            log_gam[i] = gammaln(1 + mu + i*self.x)  # use gammaln to avoid overflow
        fmy = -np.log(log_gam[0]+log_gam[2]-2*log_gam[1]) - y
        error_vector = fmy / self.ysm
        return error_vector
    
    def Vertex(self):
        """
        Get gamma fit parameters for figures that use PVS vertices.
        
        Returns:
            mu_ML: float
                mu for most likely gamma fit parameters.
            mu_Broad: float
                mu for broader PSD vertex.
            mu_Narrow: float
                mu for narrower PSD vertex.
            la_ML: float
                lambda for most likely gamma fit parameters.
            la_Broad: float
                lambda for broader PSD vertex.
            la_Narrow: float
                lambda for narrower PSD vertex.
            la_LargeD: float
                lambda for larger MMD vertex.
            la_SmallD: float
                lambda for smaller MMD vertex.
            logN0_ML: float
                log10(N0) for most likely gamma fit parameters.
            logN0_Broad: float
                log10(N0) for broader PSD vertex.
            logN0_Narrow: float
                log10(N0) for narrower PSD vertex.
            logN0_LargeD: float
                log10(N0) for larger MMD vertex.
            logN0_SmallD: float
                log10(N0) for smaller MMD vertex.
            logN0_Light:
                log10(N0) for lighter IWC vertex.
            logN0_Heavy:
                log10(N0) for heavier IWC vertex.
        """
        psi = np.linspace(0.5-0.5*chi2.cdf(self.chisq,1),0.5+0.5*chi2.cdf(self.chisq,1),3)
        Y, Z, X = np.meshgrid(norm.ppf(psi),norm.ppf(psi),norm.ppf(psi),indexing='ij')
        mu = np.zeros(Y.shape)
        mu[0][:][:] = np.min(self.mu)
        mu[1][:][:] = self.mu_ML
        mu[2][:][:] = np.max(self.mu)
        la = 0.1*np.exp((self.zm+self.zs*Z+self.a[0]*self.ys*Y)+(gammaln(1+2*self.x+mu)-gammaln(1+mu))/(2*self.x))
        logN0 = 8+((self.xm+self.xs*X+self.b[0]*self.zs*Z+self.b[1]*self.ys*Y)-gammaln(1+self.x+mu)+npm((1+self.x+mu),np.log(10*la)))/np.log(10)
        mu_ML = mu[1][1][1] #mu_m
        mu_Broad = mu[0][1][1] #mu_l
        mu_Narrow = mu[2][1][1] #mu_u
        la_ML = la[1][1][1] #la_m
        la_Broad = la[0][1][1] #la_lm
        la_Narrow = la[2][1][1] #la_um
        la_LargeD = la[1][0][1] #la_l
        la_SmallD = la[1][2][1] #la_u
        logN0_ML = logN0[1][1][1] #logN0_m
        logN0_Broad = logN0[0][1][1] #logN0_lm
        logN0_Narrow = logN0[2][1][1] #logN0_um
        logN0_LargeD = logN0[1][0][1] #logN0_ll
        logN0_SmallD = logN0[1][2][1] #logN0_ul
        logN0_Light = logN0[1][1][0] #logN0_l
        logN0_Heavy = logN0[1][1][2] #logN0_u
        return (mu_ML,mu_Broad,mu_Narrow,la_ML,la_Broad,la_Narrow,la_LargeD,la_SmallD,
                logN0_ML,logN0_Broad,logN0_Narrow,logN0_LargeD,logN0_SmallD,logN0_Light,logN0_Heavy)