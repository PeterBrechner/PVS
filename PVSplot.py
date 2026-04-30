# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:23:31 2026

@author: peter
"""

import PVS
import numpy as np
import matplotlib as mpl
import pylab
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision = 3)


def npm(ndarray1, ndarray2):
    return np.multiply(ndarray1, ndarray2)


class PlotPVS():
    def __init__(self,D,dirr,M,err,x,leg1x,leg2x,leg3x,leg4x,leg5x,leg6x,lgdx,name,chisq=4,norm=False):
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
        leg1x: array
            Indices for first quantitative regime and first categorical regime.
        leg2x: array
            Indices for second quantitative regime and first categorical regime.
        leg3x: array
            Indices for third quantitative regime and first categorical regime.
        leg4x: array
            Indices for first quantitative regime and second categorical regime.
        leg5x: array
            Indices for second quantitative regime and second categorical regime.
        leg6x: array
            Indices for third quantitative regime and second categorical regime.
        lgdx: list
            Legend for quantitative regimes.
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
        self.leg1x = leg1x
        self.leg2x = leg2x
        self.leg3x = leg3x
        self.leg4x = leg4x
        self.leg5x = leg5x
        self.leg6x = leg6x
        self.lgdx = lgdx
        self.name = name
        self.chisq = chisq
        self.norm = norm
        self.M1 = self.M[self.leg1x,:]
        self.M2 = self.M[self.leg2x,:]
        self.M3 = self.M[self.leg3x,:]
        self.M4 = self.M[self.leg4x,:]
        self.M5 = self.M[self.leg5x,:]
        self.M6 = self.M[self.leg6x,:]
        self.err1 = self.err[self.leg1x,:]
        self.err2 = self.err[self.leg2x,:]
        self.err3 = self.err[self.leg3x,:]
        self.err4 = self.err[self.leg4x,:]
        self.err5 = self.err[self.leg5x,:]
        self.err6 = self.err[self.leg6x,:]
        self.c = mpl.cm.get_cmap('viridis')
        pylab.rcParams['font.size'] = 14
    
    def BC(self):
        """
        Create a table of the Bhattacharyya coefficients for each pair of PVSs.
        """
        #Note: For common coordinates, PVSs must be normalized.
        PVSx1 = PVS.PVS(self.M1,self.err1,self.x,self.chisq,self.norm,3)
        PVSx2 = PVS.PVS(self.M2,self.err2,self.x,self.chisq,self.norm,3)
        PVSx3 = PVS.PVS(self.M3,self.err3,self.x,self.chisq,self.norm,3)
        PVSx4 = PVS.PVS(self.M4,self.err4,self.x,self.chisq,self.norm,3)
        PVSx5 = PVS.PVS(self.M5,self.err5,self.x,self.chisq,self.norm,3)
        PVSx6 = PVS.PVS(self.M6,self.err6,self.x,self.chisq,self.norm,3)
        PVSa = [PVSx1,PVSx2,PVSx3,PVSx4,PVSx5,PVSx6]
        BC = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                BC[i,j] = self.CalcBC(PVSa[i],PVSa[j])
        data = {"PVS1":[np.round(BC[0,0],3),np.round(BC[0,1],3),np.round(BC[0,2],3),np.round(BC[0,3],3),np.round(BC[0,4],3),np.round(BC[0,5],3)],
                "PVS2":[np.round(BC[1,0],3),np.round(BC[1,1],3),np.round(BC[1,2],3),np.round(BC[1,3],3),np.round(BC[1,4],3),np.round(BC[1,5],3)],
                "PVS3":[np.round(BC[2,0],3),np.round(BC[2,1],3),np.round(BC[2,2],3),np.round(BC[2,3],3),np.round(BC[2,4],3),np.round(BC[2,5],3)],
                "PVS4":[np.round(BC[3,0],3),np.round(BC[3,1],3),np.round(BC[3,2],3),np.round(BC[3,3],3),np.round(BC[3,4],3),np.round(BC[3,5],3)],
                "PVS5":[np.round(BC[4,0],3),np.round(BC[4,1],3),np.round(BC[4,2],3),np.round(BC[4,3],3),np.round(BC[4,4],3),np.round(BC[4,5],3)],
                "PVS6":[np.round(BC[5,0],3),np.round(BC[5,1],3),np.round(BC[5,2],3),np.round(BC[5,3],3),np.round(BC[5,4],3),np.round(BC[5,5],3)]}
        df = pd.DataFrame(data,index=["PVS1","PVS2","PVS3","PVS4","PVS5","PVS6"])
        df.to_excel(self.dirr+'Bhattacharyya'+self.name+'.xlsx')
    
    def CalcBC(self,PVS1,PVS2):
        """
        Calculate the Bhattacharyya coefficient for a pair of PVSs.

        Args:
            PVS1: First PVS
            PVS2: Second PVS

        Returns:
            Bhattacharyya coefficient
        """
        
        if self.norm:
            #Use closed form expression for pairs of normalized PVSs. 
            yp = PVS2.ym-PVS1.ym
            zp = PVS2.zm-PVS1.zm
            xp = PVS2.xm-PVS1.xm
            chi2y = 0.5*yp**2/(PVS1.ys**2+PVS2.ys**2)
            chi2z = 0.5*zp**2/(PVS1.zs**2+PVS2.zs**2)
            chi2x = 0.5*xp**2/(PVS1.xs**2+PVS2.xs**2)
            num = np.sqrt((2*PVS1.ys*PVS2.ys)*(2*PVS1.zs*PVS2.zs)*(2*PVS1.xs*PVS2.xs))
            den = np.sqrt((PVS1.ys**2+PVS2.ys**2)*(PVS1.zs**2+PVS2.zs**2)*(PVS1.xs**2+PVS2.xs**2))
            return (num/den)*np.exp(-0.5*(chi2y+chi2z+chi2x))
        else:
            #No closed form expression exists for the general case.
            #Must sum the geometric mean of probability densities over an (x,y,z) grid to estimate the triple integral.
            #Optimize the (x,y,z) grid to minimize numerical error.
            xm = (PVS1.xm*PVS2.xs**2+PVS2.xm*PVS1.xs**2)/(PVS1.xs**2+PVS2.xs**2)
            ym = (PVS1.ym*PVS2.ys**2+PVS2.ym*PVS1.ys**2)/(PVS1.ys**2+PVS2.ys**2)
            zm = (PVS1.zm*PVS2.zs**2+PVS2.zm*PVS1.zs**2)/(PVS1.zs**2+PVS2.zs**2)
            stdx1 = np.sqrt(PVS1.xs**2+PVS1.b[0]**2*PVS1.zs**2+PVS1.b[1]**2*PVS1.ys**2)
            stdx2 = np.sqrt(PVS2.xs**2+PVS2.b[0]**2*PVS2.zs**2+PVS2.b[1]**2*PVS2.ys**2)
            stdz1 = np.sqrt(PVS1.zs**2+PVS1.a[0]**2*PVS1.ys**2)
            stdz2 = np.sqrt(PVS2.zs**2+PVS2.a[0]**2*PVS2.ys**2)
            x = np.linspace(xm-4*np.sqrt(stdx1*stdx2),xm+4*np.sqrt(stdx1*stdx2),33)
            y = np.linspace(ym-4*np.sqrt(PVS1.ys*PVS2.ys),ym+4*np.sqrt(PVS1.ys*PVS2.ys),33)
            z = np.linspace(zm-4*np.sqrt(stdz1*stdz2),zm+4*np.sqrt(stdz1*stdz2),33)
            #Compute the geometric mean of probability densities at each point on the (x,y,z) grid.
            den = np.sqrt((2*np.pi*PVS1.xs*PVS2.xs)*(2*np.pi*PVS1.ys*PVS2.ys)*(2*np.pi*PVS1.zs*PVS2.zs))
            prob = np.zeros((len(x),len(y),len(z)))
            for i in range(len(x)):
                for j in range(len(y)):
                    for k in range(len(z)):
                        prob[i,j,k] = (1/den)*np.exp(-0.25*(PVS1.chi2x(y[j],z[k],x[i])**2+PVS2.chi2x(y[j],z[k],x[i])**2
                                                            +PVS1.chi2y(y[j])**2+PVS2.chi2y(y[j])**2
                                                            +PVS1.chi2z(y[j],z[k])**2+PVS2.chi2z(y[j],z[k])**2))
            #Multiply the sum of the probability densities by the grid spacing.
            return np.sum(prob)*(x[1]-x[0])*(y[1]-y[0])*(z[1]-z[0])
    
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
        PVSx = PVS.PVS(self.M1,self.err1,self.x,self.chisq,self.norm,3)
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = PVSx.Vertex()
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
        PVSx = PVS.PVS(self.M1,self.err1,self.x,self.chisq,self.norm,3)
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = PVSx.Vertex()
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
        PVSx = PVS.PVS(self.M1,self.err1,self.x,self.chisq,self.norm,3)
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = PVSx.Vertex()
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
        PVSx = PVS.PVS(self.M1,self.err1,self.x,self.chisq,self.norm,3)
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = PVSx.Vertex()
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
        PVSx = PVS.PVS(self.M1,self.err1,self.x,self.chisq,self.norm,3)
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = PVSx.Vertex()
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

    def VertexParams(self,cat1,cat2):
        """
        Creates table of gamma fit parameters for the vertices and most likely solution of a PVS.
        Outputs linear regression coefficients to JSON file.
        
        Args:
            cat1: Name of first categorical regime.
            cat2: Name of second categorical regime.
        """
        PVSx1 = PVS.PVS(self.M1,self.err1,self.x,self.chisq,self.norm,3)
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = PVSx1.Vertex()
        PVSx2 = PVS.PVS(self.M2,self.err2,self.x,self.chisq,self.norm,3)
        [mu_m2,mu_l2,mu_u2,la_m2,la_lm2,la_um2,la_l2,la_u2,logN0_m2,logN0_lm2,
         logN0_um2,logN0_ll2,logN0_ul2,logN0_l2,logN0_u2] = PVSx2.Vertex()
        PVSx3 = PVS.PVS(self.M3,self.err3,self.x,self.chisq,self.norm,3)
        [mu_m3,mu_l3,mu_u3,la_m3,la_lm3,la_um3,la_l3,la_u3,logN0_m3,logN0_lm3,
         logN0_um3,logN0_ll3,logN0_ul3,logN0_l3,logN0_u3] = PVSx3.Vertex()
        PVSx4 = PVS.PVS(self.M4,self.err4,self.x,self.chisq,self.norm,3)
        [mu_m4,mu_l4,mu_u4,la_m4,la_lm4,la_um4,la_l4,la_u4,logN0_m4,logN0_lm4,
         logN0_um4,logN0_ll4,logN0_ul4,logN0_l4,logN0_u4] = PVSx4.Vertex()
        PVSx5 = PVS.PVS(self.M5,self.err5,self.x,self.chisq,self.norm,3)
        [mu_m5,mu_l5,mu_u5,la_m5,la_lm5,la_um5,la_l5,la_u5,logN0_m5,logN0_lm5,
         logN0_um5,logN0_ll5,logN0_ul5,logN0_l5,logN0_u5] = PVSx5.Vertex()
        PVSx6 = PVS.PVS(self.M6,self.err6,self.x,self.chisq,self.norm,3)
        [mu_m6,mu_l6,mu_u6,la_m6,la_lm6,la_um6,la_l6,la_u6,logN0_m6,logN0_lm6,
         logN0_um6,logN0_ll6,logN0_ul6,logN0_l6,logN0_u6] = PVSx6.Vertex()
        data = {"Category":[self.lgdx[0],self.lgdx[1],self.lgdx[2]],
                "PVS Vertex (down)":[r'(log$_{10} N_0$, $\mu$, $\lambda$)',
                                     r'(log$_{10} N_0$ ,$\mu$, $\lambda$)',
                                     r'(log$_{10} N_0$, $\mu$, $\lambda$)'],
                "Most Likely, "+cat1:[(float(np.round(logN0_m1,2)),float(np.round(mu_m1,2)),float(np.round(la_m1,2))),
                                      (float(np.round(logN0_m2,2)),float(np.round(mu_m2,2)),float(np.round(la_m2,2))),
                                      (float(np.round(logN0_m3,2)),float(np.round(mu_m3,2)),float(np.round(la_m3,2)))],
                "Lighter "+cat1:[(float(np.round(logN0_l1,2)),float(np.round(mu_m1,2)),float(np.round(la_m1,2))),
                                      (float(np.round(logN0_l2,2)),float(np.round(mu_m2,2)),float(np.round(la_m2,2))),
                                      (float(np.round(logN0_l3,2)),float(np.round(mu_m3,2)),float(np.round(la_m3,2)))],
                "Heavier "+cat1:[(float(np.round(logN0_u1,2)),float(np.round(mu_m1,2)),float(np.round(la_m1,2))),
                                      (float(np.round(logN0_u2,2)),float(np.round(mu_m2,2)),float(np.round(la_m2,2))),
                                      (float(np.round(logN0_u3,2)),float(np.round(mu_m3,2)),float(np.round(la_m3,2)))],
                "Broader "+cat1+" PSD":[(float(np.round(logN0_lm1,2)),float(np.round(mu_l1,2)),float(np.round(la_lm1,2))),
                                      (float(np.round(logN0_lm2,2)),float(np.round(mu_l2,2)),float(np.round(la_lm2,2))),
                                      (float(np.round(logN0_lm3,2)),float(np.round(mu_l3,2)),float(np.round(la_lm3,2)))],
                "Narrower "+cat1+" PSD":[(float(np.round(logN0_um1,2)),float(np.round(mu_u1,2)),float(np.round(la_um1,2))),
                                      (float(np.round(logN0_um2,2)),float(np.round(mu_u2,2)),float(np.round(la_um2,2))),
                                      (float(np.round(logN0_um3,2)),float(np.round(mu_u3,2)),float(np.round(la_um3,2)))],
                "Larger "+cat1+" MMD":[(float(np.round(logN0_ll1,2)),float(np.round(mu_m1,2)),float(np.round(la_l1,2))),
                                      (float(np.round(logN0_ll2,2)),float(np.round(mu_m2,2)),float(np.round(la_l2,2))),
                                      (float(np.round(logN0_ll3,2)),float(np.round(mu_m3,2)),float(np.round(la_l3,2)))],
                "Smaller "+cat1+" MMD":[(float(np.round(logN0_ul1,2)),float(np.round(mu_m1,2)),float(np.round(la_u1,2))),
                                      (float(np.round(logN0_ul2,2)),float(np.round(mu_m2,2)),float(np.round(la_u2,2))),
                                      (float(np.round(logN0_ul3,2)),float(np.round(mu_m3,2)),float(np.round(la_u3,2)))],
                "Most Likely, "+cat2:[(float(np.round(logN0_m4,2)),float(np.round(mu_m4,2)),float(np.round(la_m4,2))),
                                      (float(np.round(logN0_m5,2)),float(np.round(mu_m5,2)),float(np.round(la_m5,2))),
                                      (float(np.round(logN0_m6,2)),float(np.round(mu_m6,2)),float(np.round(la_m6,2)))],
                "Lighter "+cat2:[(float(np.round(logN0_l4,2)),float(np.round(mu_m4,2)),float(np.round(la_m4,2))),
                                      (float(np.round(logN0_l5,2)),float(np.round(mu_m5,2)),float(np.round(la_m5,2))),
                                      (float(np.round(logN0_l6,2)),float(np.round(mu_m6,2)),float(np.round(la_m6,2)))],
                "Heavier "+cat2:[(float(np.round(logN0_u4,2)),float(np.round(mu_m4,2)),float(np.round(la_m4,2))),
                                      (float(np.round(logN0_u5,2)),float(np.round(mu_m5,2)),float(np.round(la_m5,2))),
                                      (float(np.round(logN0_u6,2)),float(np.round(mu_m6,2)),float(np.round(la_m6,2)))],
                "Broader "+cat2+" PSD":[(float(np.round(logN0_lm4,2)),float(np.round(mu_l4,2)),float(np.round(la_lm4,2))),
                                      (float(np.round(logN0_lm5,2)),float(np.round(mu_l5,2)),float(np.round(la_lm5,2))),
                                      (float(np.round(logN0_lm6,2)),float(np.round(mu_l6,2)),float(np.round(la_lm6,2)))],
                "Narrower "+cat2+" PSD":[(float(np.round(logN0_um4,2)),float(np.round(mu_u4,2)),float(np.round(la_um4,2))),
                                      (float(np.round(logN0_um5,2)),float(np.round(mu_u5,2)),float(np.round(la_um5,2))),
                                      (float(np.round(logN0_um6,2)),float(np.round(mu_u6,2)),float(np.round(la_um6,2)))],
                "Larger "+cat2+" MMD":[(float(np.round(logN0_ll4,2)),float(np.round(mu_m4,2)),float(np.round(la_l4,2))),
                                      (float(np.round(logN0_ll5,2)),float(np.round(mu_m5,2)),float(np.round(la_l5,2))),
                                      (float(np.round(logN0_ll6,2)),float(np.round(mu_m6,2)),float(np.round(la_l6,2)))],
                "Smaller "+cat2+" MMD":[(float(np.round(logN0_ul4,2)),float(np.round(mu_m4,2)),float(np.round(la_u4,2))),
                                      (float(np.round(logN0_ul5,2)),float(np.round(mu_m5,2)),float(np.round(la_u5,2))),
                                      (float(np.round(logN0_ul6,2)),float(np.round(mu_m6,2)),float(np.round(la_u6,2)))]}
        df = pd.DataFrame(data)
        fig,ax = pylab.subplots(dpi=150,figsize=(6.4,4.8))
        ax.axis('off')
        colors = np.zeros((16,3),dtype='U7')
        colors[0,:] = '#ffffff'
        colors[1,:] = '#cccccc'
        colors[2,:] = '#ffffff'
        colors[3:5,:] = '#ffffcc'
        colors[5:7,:] = '#ccffff'
        colors[7:9,:] = '#ffccff'
        colors[9,:] = '#ffffff'
        colors[10:12,:] = '#ffffcc'
        colors[12:14,:] = '#ccffff'
        colors[14:16,:] = '#ffccff'
        colors2 = np.zeros(16,dtype='U7')
        colors2[0] = '#ffffff'
        colors2[1] = '#cccccc'
        colors2[2] = '#ffffff'
        colors2[3:5] = '#ffffcc'
        colors2[5:7] = '#ccffff'
        colors2[7:9] = '#ffccff'
        colors2[9] = '#ffffff'
        colors2[10:12] = '#ffffcc'
        colors2[12:14] = '#ccffff'
        colors2[14:16] = '#ffccff'
        table = ax.table(cellText=df.T.values,cellColours=colors,rowColours=colors2,rowLabels=df.columns,loc='center')
        table_props = table.properties()
        table_cells = table_props['children']
        table._autoColumns = []
        for cell in table_cells:
            cell.set_width(0.30)
            cell.set_height(0.06)
            cell.PAD = 0.03
        fig.set_tight_layout(True)
        pylab.savefig(self.dirr+'VertexParams'+self.name+'.png')
        a_0 = [PVSx1.a[0],PVSx2.a[0],PVSx3.a[0],PVSx4.a[0],PVSx5.a[0],PVSx6.a[0]]
        b_0 = [PVSx1.b[0],PVSx2.b[0],PVSx3.b[0],PVSx4.b[0],PVSx5.b[0],PVSx6.b[0]]
        b_1 = [PVSx1.b[1],PVSx2.b[1],PVSx3.b[1],PVSx4.b[1],PVSx5.b[1],PVSx6.b[1]]
        with open(self.dirr+'VertexParams'+self.name+'.txt','w') as f:
            print(json.dumps('z vs. y (when positive, MMD decreases as mu increases)'),file=f)
            print(json.dumps(a_0),file=f)
            print(json.dumps('x vs. z (when positive, IWC decreases as MMD increases)'),file=f)
            print(json.dumps(b_0),file=f)
            print(json.dumps('x vs. y (when positive, IWC increases as mu increases)'),file=f)
            print(json.dumps(b_1),file=f)

    def VertexFitSingle(self):
        """
        Plots N(D) for the vertices and most likely solution of a single PVS in a figure saved outside function call.
        """
        PVSx = PVS.PVS(self.M1,self.err1,self.x,self.chisq,self.norm,3)
        [mu_m1,mu_l1,mu_u1,la_m1,la_lm1,la_um1,la_l1,la_u1,logN0_m1,logN0_lm1,
         logN0_um1,logN0_ll1,logN0_ul1,logN0_l1,logN0_u1] = PVSx.Vertex()
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

    def VolCrossCompare(self,M,err,color):
        """
        Plots features of a PVS for illustration in a figure initialized outside function call.
        
        Args:
            M: Values of self.M used
            err: Values of self.err used
            color: Color used for PVS graphic.
        """
        PVSx = PVS.PVS(M,err,self.x,self.chisq,self.norm)
        var = PVSx.Plot(color)
        y = var[0]
        z = var[1]
        y2 = var[2]
        z2l = var[3]
        z2u = var[4]
        z2a = z2l[np.where(np.abs(y2-y) == np.min(np.abs(y2-y)))]
        z2b = z2u[np.where(np.abs(y2-y) == np.min(np.abs(y2-y)))]
        pylab.scatter(y,z,25,color=[0,0,0],zorder=2.5)
        pylab.scatter(-2,-1,25,color=color,label=self.lgdx[0]) # dummy point for legend
        pylab.vlines(y,z2a,z2b,color='k')
        pylab.plot(y2,np.sqrt(npm(z2l,z2u)),color=self.c(1.0))

    def VolCrossProjection(self,M,err,jumpup=True):
        """
        Projects the surface of a 3D PVS onto the mu-lambda axes. Fill color varies with N0.
        Opposite quarters of the 3D surface are projected onto top and bottom halves of the 
        cross-section to visualize the thickness of the 3D PVS.
        
        Args:
            M: Values of self.M used.
            err: Values of self.err used.
            jumpup: Boolean, optional
                If True, projection of N0 onto mu-lambda cross-section jumps up 
                as lambda increases across its most likely value.
                If False, projection of N0 onto mu-lambda cross-section jumps down 
                as lambda increases across its most likely value.
        """
        fig = pylab.figure(figsize=(6,4.8),dpi=150)
        PVSx = PVS.PVS(M,err,self.x,self.chisq,self.norm)
        var = PVSx.PVS(jumpup)
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

    def VolCrossSingle(self,M,err,color):
        """
        Like VolCrossCompare, but initializes figure and adds a fill color that varies with N0.
        
        Args:
            M: Values of self.M used.
            err: Values of self.err used.
            color: Color used for PVS graphic.
        """
        fig = pylab.figure(figsize=(6,4.8),dpi=150)
        PVSx = PVS.PVS(M,err,self.x,self.chisq,self.norm)
        var = PVSx.Plot(color,False,True)
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

    def VolCrossSix(self,unit=""):
        """
        Plots six PVS cross-sections in one figure.
        
        Args:
            unit: Unit for figure title. The default is "".
        """
        fig = pylab.figure(figsize=(7,7),dpi=150)
        PVSx1 = PVS.PVS(self.M1,self.err1,self.x,self.chisq,self.norm)
        var = PVSx1.Plot(self.c(0))
        x1 = var[0]
        y1 = var[1]
        PVSx2 = PVS.PVS(self.M2,self.err2,self.x,self.chisq,self.norm)
        var = PVSx2.Plot(self.c(0.4))
        x2 = var[0]
        y2 = var[1]
        PVSx3 = PVS.PVS(self.M3,self.err3,self.x,self.chisq,self.norm)
        var = PVSx3.Plot(self.c(0.8))
        x3 = var[0]
        y3 = var[1]
        PVSx4 = PVS.PVS(self.M4,self.err4,self.x,self.chisq,self.norm)
        var = PVSx4.Plot(self.c(0),True)
        x4 = var[0]
        y4 = var[1]
        PVSx5 = PVS.PVS(self.M5,self.err5,self.x,self.chisq,self.norm)
        var = PVSx5.Plot(self.c(0.4),True)
        x5 = var[0]
        y5 = var[1]
        PVSx6 = PVS.PVS(self.M6,self.err6,self.x,self.chisq,self.norm)
        var = PVSx6.Plot(self.c(0.8),True)
        x6 = var[0]
        y6 = var[1]
        pylab.scatter(x1,y1,25,color=self.c(0),zorder=2.5,label=self.lgdx[0])
        pylab.scatter(x2,y2,25,color=self.c(0.4),zorder=2.5,label=self.lgdx[1])
        pylab.scatter(x3,y3,25,color=self.c(0.8),zorder=2.5,label=self.lgdx[2])
        pylab.scatter(x4,y4,25,color=self.c(0),zorder=2.5,marker='^')
        pylab.scatter(x5,y5,25,color=self.c(0.4),zorder=2.5,marker='^')
        pylab.scatter(x6,y6,25,color=self.c(0.8),zorder=2.5,marker='^')
        pylab.xlabel(r"$\mu$")
        pylab.ylabel(r"$\lambda$ [mm$^{-1}$]")
        pylab.xlim([-1,11])
        pylab.ylim([0,25])
        fig.legend(loc='upper center',bbox_to_anchor=(0.5,0.935),ncol=3)
        pylab.title("Cross Sections for "+self.name+unit+"\n\n")
        fig.set_tight_layout(True)
        pylab.savefig(self.dirr+'VolCrossSix'+self.name+'.png')