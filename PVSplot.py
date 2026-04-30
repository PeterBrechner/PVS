# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:23:31 2026

@author: peter
"""

import PVS
import numpy as np
import matplotlib as mpl
import pylab
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision = 3)


def npm(ndarray1, ndarray2):
    return np.multiply(ndarray1, ndarray2)


class PlotPVS():
    def __init__(self,D,dirr,M,err,x,name,chisq=4,norm=False):
        """
        Plot PVS objects for three quantitative regimes (e.g., colder, moderate, warmer)
        in two categorical regimes (e.g., ice phase, mixed phase).
        
        Parameters
        ----------
        D: array, in mm
            1e3 times self.D[idx] from PVSgamma input.
        dirr: str
            Plot directory.
        M: 2d array
            M from PVSgamma output MAT file.
        err: 2d array
            err from PVSgamma output MAT file.
        x: float
            x from PVSgamma output MAT file.
        name: str
            Name for plots.
        chisq: float, optional
            chi squared. The default is 4.
        norm: Boolean, optional
            Normalized PVSs if True, real PVSs if False. The default is False.
        """
        self.D = D
        self.dirr = dirr
        self.M = M
        self.err = err
        self.x = x
        self.name = name
        self.chisq = chisq
        self.norm = norm
        self.PVSx = PVS.PVS(self.M,self.err,self.x,self.chisq,self.norm)
        self.c = mpl.cm.get_cmap('viridis')
        pylab.rcParams['font.size'] = 14
    
    
    def NofD(self,logN0,mu,la):
        """
        Calculate size distribution N(D) for gamma fit parameters N0, mu, lambda.

        Args:
            logN0: Values of log10(N0).
            mu: Values of mu.
            la: Values of lambda.
        
        Returns:
            Size distribution N(D) in m^-3 mm^-1 for gamma fit parameters N0, mu, lambda.
        """
        return 1e-3*npm(npm(10**logN0,np.power(0.1*self.D,mu)),np.exp(-npm(0.1*self.D,10*la)))

    def VertexFit(self):
        """
        Plots N(D) for the vertices and most likely solution of a PVS.
        """
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = self.PVSx.Vertex()
        fig,ax = pylab.subplots(figsize=(6.4,4.8),dpi=150)
        pylab.loglog(self.D,self.NofD(logN0_m1,mu_m1,la_m1),color='k',lw=3,
                     label=r"ML (most likely)")
        pylab.loglog(self.D,self.NofD(logN0_l1,mu_m1,la_m1),'--',color=self.c(0.0),lw=2,
                     label=r"$N_0 < N_{0,ML}$ (lighter)")
        pylab.loglog(self.D,self.NofD(logN0_u1,mu_m1,la_m1),color=self.c(0.0),lw=2,
                     label=r"$N_0 > N_{0,ML}$ (heavier)")
        pylab.loglog(self.D,self.NofD(logN0_lm1,mu_l1,la_lm1),'--',color=self.c(0.8),lw=2,zorder=2.5,
                     label=r"$\mu < \mu_{ML}$ (broader PSD)")
        pylab.loglog(self.D,self.NofD(logN0_um1,mu_u1,la_um1),color=self.c(0.8),lw=2,zorder=2.5,
                     label=r"$\mu > \mu_{ML}$ (narrower PSD)")
        pylab.loglog(self.D,self.NofD(logN0_ll1,mu_m1,la_l1),'--',color=self.c(0.4),lw=2,zorder=2.25,
                     label=r"$\lambda < \lambda_{ML}$ (larger MMD)")
        pylab.loglog(self.D,self.NofD(logN0_ul1,mu_m1,la_u1),color=self.c(0.4),lw=2,zorder=2.25,
                     label=r"$\lambda > \lambda_{ML}$ (smaller MMD)")
        pylab.title(r'Gamma Fits to $N$($D$) vs. $D$')
        pylab.xlabel(r'$D$ [mm]')
        pylab.ylabel(r'$N$($D$) [m$^{-3}$ mm$^{-1}$]')
        pylab.legend(fontsize=12)
        pylab.xlim(1e-1,5e1)
        pylab.ylim(1e1,1e6)
        fig.set_tight_layout(True)
        pylab.savefig(self.dirr+'VertexFit'+self.name+'.png')
        
    def VertexFitA(self):
        """
        Plots N(D) for the most likely solution of a PVS.
        """
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = self.PVSx.Vertex()
        fig,ax = pylab.subplots(figsize=(6.4,4.8),dpi=150)
        pylab.loglog(self.D,self.NofD(logN0_m1,mu_m1,la_m1),color='k',lw=3,
                     label=r"ML (most likely)")
        pylab.title(r'Gamma Fits to $N$($D$) vs. $D$')
        pylab.xlabel(r'$D$ [mm]')
        pylab.ylabel(r'$N$($D$) [m$^{-3}$ mm$^{-1}$]')
        pylab.legend(fontsize=12)
        pylab.xlim(1e-1,2e1)
        pylab.ylim(1e1,1e5)
        fig.set_tight_layout(True)
        pylab.savefig(self.dirr+'VertexFitA'+self.name+'.png')
    
    def VertexFitB(self):
        """
        Plots N(D) for the vertices in mu and most likely solution of a PVS.
        """
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = self.PVSx.Vertex()
        fig,ax = pylab.subplots(figsize=(6.4,4.8),dpi=150)
        pylab.loglog(self.D,self.NofD(logN0_m1,mu_m1,la_m1),color='k',lw=3,
                     label=r"ML (most likely)")
        pylab.loglog(self.D,self.NofD(logN0_lm1,mu_l1,la_lm1),'--',color=self.c(0.4),lw=2,zorder=2.5,
                     label=r"$\mu < \mu_{ML}$ (broader)")
        pylab.loglog(self.D,self.NofD(logN0_um1,mu_u1,la_um1),color=self.c(0.4),lw=2,zorder=2.5,
                     label=r"$\mu > \mu_{ML}$ (narrower)")
        pylab.title(r'Gamma Fits to $N$($D$) vs. $D$')
        pylab.xlabel(r'$D$ [mm]')
        pylab.ylabel(r'$N$($D$) [m$^{-3}$ mm$^{-1}$]')
        pylab.legend(fontsize=12)
        pylab.xlim(1e-1,2e1)
        pylab.ylim(1e1,1e5)
        fig.set_tight_layout(True)
        pylab.savefig(self.dirr+'VertexFitB'+self.name+'.png')
        
    def VertexFitC(self):
        """
        Plots N(D) for the vertices in lambda and most likely solution of a PVS.
        """
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = self.PVSx.Vertex()
        fig,ax = pylab.subplots(figsize=(6.4,4.8),dpi=150)
        pylab.loglog(self.D,self.NofD(logN0_m1,mu_m1,la_m1),color='k',lw=3,
                     label=r"ML (most likely)")
        pylab.loglog(self.D,self.NofD(logN0_ll1,mu_m1,la_l1),'--',color=self.c(0.4),lw=2,zorder=2.25,
                     label=r"$\lambda < \lambda_{ML}$ (larger $D$)")
        pylab.loglog(self.D,self.NofD(logN0_ul1,mu_m1,la_u1),color=self.c(0.4),lw=2,zorder=2.25,
                     label=r"$\lambda > \lambda_{ML}$ (smaller $D$)")
        pylab.title(r'Gamma Fits to $N$($D$) vs. $D$')
        pylab.xlabel(r'$D$ [mm]')
        pylab.ylabel(r'$N$($D$) [m$^{-3}$ mm$^{-1}$]')
        pylab.legend(fontsize=12)
        pylab.xlim(1e-1,2e1)
        pylab.ylim(1e1,1e5)
        fig.set_tight_layout(True)
        pylab.savefig(self.dirr+'VertexFitC'+self.name+'.png')
    
    def VertexFitD(self):
        """
        Plots N(D) for the vertices in N0 and most likely solution of a PVS.
        """
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = self.PVSx.Vertex()
        fig,ax = pylab.subplots(figsize=(6.4,4.8),dpi=150)
        pylab.loglog(self.D,self.NofD(logN0_m1,mu_m1,la_m1),color='k',lw=3,
                     label=r"ML (most likely)")
        pylab.loglog(self.D,self.NofD(logN0_l1,mu_m1,la_m1),'--',color=self.c(0.4),lw=2,
                     label=r"$N_0 < N_{0,ML}$ (lighter)")
        pylab.loglog(self.D,self.NofD(logN0_u1,mu_m1,la_m1),color=self.c(0.4),lw=2,
                     label=r"$N_0 > N_{0,ML}$ (heavier)")
        pylab.title(r'Gamma Fits to $N$($D$) vs. $D$')
        pylab.xlabel(r'$D$ [mm]')
        pylab.ylabel(r'$N$($D$) [m$^{-3}$ mm$^{-1}$]')
        pylab.legend(fontsize=12)
        pylab.xlim(1e-1,2e1)
        pylab.ylim(1e1,1e5)
        fig.set_tight_layout(True)
        pylab.savefig(self.dirr+'VertexFitD'+self.name+'.png')

    def VertexFitSingle(self):
        """
        Plots N(D) for the vertices and most likely solution of a single PVS in a figure saved outside function call.
        """
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = self.PVSx.Vertex()
        fig,ax = pylab.subplots(figsize=(6.4,4.8),dpi=150)
        pylab.loglog(self.D,self.NofD(logN0_m1,mu_m1,la_m1),color=self.c(0.8),lw=2,zorder=2.25)
        pylab.loglog(self.D,self.NofD(logN0_l1,mu_m1,la_m1),color=self.c(0.8),lw=2,zorder=2.25)
        pylab.loglog(self.D,self.NofD(logN0_u1,mu_m1,la_m1),color=self.c(0.8),lw=2,zorder=2.25)
        pylab.loglog(self.D,self.NofD(logN0_lm1,mu_l1,la_lm1),color=self.c(0.8),lw=2,zorder=2.25)
        pylab.loglog(self.D,self.NofD(logN0_um1,mu_u1,la_um1),color=self.c(0.8),lw=2,zorder=2.25)
        pylab.loglog(self.D,self.NofD(logN0_ll1,mu_m1,la_l1),color=self.c(0.8),lw=2,zorder=2.25)
        pylab.loglog(self.D,self.NofD(logN0_ul1,mu_m1,la_u1),color=self.c(0.8),lw=2,zorder=2.25)
        pylab.xlabel(r'$D$ [mm]')
        pylab.ylabel(r'$N$($D$) [m$^{-3}$ mm$^{-1}$]')
        pylab.legend(fontsize=12)
        pylab.xlim(1e-1,2e1)
        pylab.ylim(1e0,1e4)
        fig.set_tight_layout(True)

    def VolCrossCompare(self,color,lgdx):
        """
        Plots features of a PVS for illustration in a figure initialized outside function call.
                
        Args:
            color: Color used for PVS graphic.
            lgdx: Legend entry.
        """
        var = self.PVSx.Plot(color)
        y = var[0]
        z = var[1]
        y2 = var[2]
        z2l = var[3]
        z2u = var[4]
        z2a = z2l[np.where(np.abs(y2-y) == np.min(np.abs(y2-y)))]
        z2b = z2u[np.where(np.abs(y2-y) == np.min(np.abs(y2-y)))]
        pylab.scatter(y,z,25,color=[0,0,0],zorder=2.5)
        pylab.scatter(-2,-1,25,color=color,label=lgdx) # dummy point for legend
        pylab.vlines(y,z2a,z2b,color='k')
        pylab.plot(y2,np.sqrt(npm(z2l,z2u)),color=self.c(1.0))

    def VolCrossProjection(self,jumpup=True):
        """
        Projects the surface of a 3D PVS onto the mu-lambda axes. Fill color varies with N0.
        Opposite quarters of the 3D surface are projected onto top and bottom halves of the 
        cross-section to visualize the thickness of the 3D PVS.
        
        Args:
            jumpup: Boolean, optional
                If True, projection of N0 onto mu-lambda cross-section jumps up 
                as lambda increases across its most likely value.
                If False, projection of N0 onto mu-lambda cross-section jumps down 
                as lambda increases across its most likely value.
        """
        fig = pylab.figure(figsize=(6,4.8),dpi=150)
        var = self.PVSx.PVS(jumpup)
        y = var[1]
        z = var[2]
        X = var[3]
        Y = var[4]
        Z = var[5]
        pylab.scatter(y,z,25,color=[1,1,1],zorder=2.5)
        pylab.scatter(Y,Z,self.chisq,c=X)
        pylab.colorbar()
        pylab.xlim(left=-1)
        pylab.ylim(bottom=0)
        pylab.xlabel(r"$\mu$")
        pylab.ylabel(r"$\lambda$ [mm$^{-1}$]")
        fig.set_tight_layout(True)
        pylab.savefig(self.dirr+'VolCrossProjection'+self.name+'.png')

    def VolCrossSingle(self,color):
        """
        Like VolCrossCompare, but initializes figure and adds a fill color that varies with N0.
        
        Args:
            color: Color used for PVS graphic.
        """
        fig = pylab.figure(figsize=(6,4.8),dpi=150)
        var = self.PVSx.Plot(color,False,True)
        y = var[0]
        z = var[1]
        y2 = var[2]
        z2l = var[3]
        z2u = var[4]
        X = var[5]
        Y = var[6]
        Z = var[7]
        z2a = z2l[np.where(np.abs(y2-y) == np.min(np.abs(y2-y)))]
        z2b = z2u[np.where(np.abs(y2-y) == np.min(np.abs(y2-y)))]
        pylab.scatter(y,z,25,color=[1,1,1],zorder=2.5)
        pylab.vlines(y,z2a,z2b,color=self.c(0.9))
        pylab.plot(y2,np.sqrt(npm(z2l,z2u)),color=[0,0,0])
        pylab.scatter(Y,Z,self.chisq,c=X)
        pylab.colorbar()
        pylab.xlim(left=-1)
        pylab.ylim(bottom=0)
        pylab.xlabel(r"$\mu$")
        pylab.ylabel(r"$\lambda$ [mm$^{-1}$]")
        fig.set_tight_layout(True)
        pylab.savefig(self.dirr+'VolCrossSingle'+self.name+'.png')