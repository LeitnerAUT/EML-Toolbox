import numpy as np;
from scipy import optimize as sp_opt;
from numpy.polynomial.legendre import legfit as LegendreFit;
from numpy.polynomial.legendre import legval as LegendreEval;
import matplotlib.pyplot as plt;

#%% define functions:
def R(coeff,Theta):
    return LegendreEval(np.cos(Theta),coeff);
    
def coeff_R(varTheta,varRadii):
    # ToDo (for plotting): optimize LegendreFit for multiple fits at once: just 1D x-Axis, but two dimensional y axis
    return LegendreFit(np.cos(varTheta),varRadii,6);
    
# function for declination angle optimization:    
def L_TestTheta(dTheta,varTheta,varRadii):
    return np.sqrt(np.sum( (varRadii - R( coeff_R(varTheta + dTheta,varRadii),varTheta + dTheta) )**2 ));
    
# functions for optimizing location of origin, search direction perpendicular to symetry axis:
def Theta_TestX(dx,varTheta,varRadii):
    return np.arctan2(np.sin(varTheta)+dx/varRadii,np.cos(varTheta));
    
def Radii_TestX(dx,varTheta,varRadii):
    return varRadii*np.sqrt( (np.sin(varTheta)+dx/varRadii)**2 + np.cos(varTheta)**2);
    
def L_TestX(dx,varTheta,varRadii,varThetaEval):
    R_Fit = R(coeff_R(Theta_TestX(dx,varTheta,varRadii),Radii_TestX(dx,varTheta,varRadii)),varThetaEval);
    return np.sqrt(np.sum( (Radii_TestX(dx,varTheta,varRadii) - R_Fit)**2 ));
    
# functions for optimizing location of origin, search direction along symetry axis:
def Theta_TestY(dy,varTheta,varRadii):
    return np.arctan2(np.sin(varTheta),np.cos(varTheta) + dy/varRadii);
    
def Radii_TestY(dy,varTheta,varRadii):
    return varRadii*np.sqrt( np.sin(varTheta)**2 + (np.cos(varTheta) + dy/varRadii)**2);
    
def L_TestY(dy,varTheta,varRadii,varThetaEval):
    R_Fit = R(coeff_R(Theta_TestY(dy,varTheta,varRadii),Radii_TestY(dy,varTheta,varRadii)),varThetaEval);
    return np.sqrt(np.sum( (Radii_TestY(dy,varTheta,varRadii) - R_Fit)**2 ));    

# main fit routine:    
def FitRadii(Data_Theta, Data_Radii, ThetaOptLim_Rad = (-1/2*np.pi, 1/2*np.pi), debug = False):
    
    if debug:
        plt.figure();
        plt.polar(Data_Theta,np.hstack( (Data_Radii, Data_Radii[1])));
        plt.title("Raw data")
    
    #%% prepare data for fit:
    Theta = Data_Theta - np.pi;
    
    # remap data from 0->2pi to -pi to pi for legendre fitting:
    Radii_Fit = np.hstack( (np.flipud(Data_Radii[0:271]),
                            np.flipud(Data_Radii[271:360]),
                            Data_Radii[271]) );
    
    if debug:
        Theta_Plot = Theta[::-1] + np.pi/2;
        
        # should look identical to first figure!
        plt.figure();
        plt.polar(Theta_Plot,Radii_Fit);
        plt.title(r"Remapped data (from $0 - 2\,\pi$ to $-\pi - +\pi$)")
    
    #%% find declination angle:
    if not ThetaOptLim_Rad[0] == ThetaOptLim_Rad[1]:
        ThetaOpt = sp_opt.minimize_scalar(L_TestTheta,
                                          args=(Theta,Radii_Fit),
                                          method='Bounded',
                                          bounds=(ThetaOptLim_Rad[0], ThetaOptLim_Rad[1])).x;
    else:
        ThetaOpt = 0;
    
    if debug:
        
        dTheta_Vector = np.linspace(-np.pi,np.pi,1000);
        L_Theta = np.zeros_like(dTheta_Vector);
        
        it = np.nditer(dTheta_Vector,flags=['f_index']);
        for dTheta in it:
            L_Theta[it.index] = L_TestTheta(dTheta,Theta,Radii_Fit);
            
        
        plt.figure();
        plt.plot(dTheta_Vector,L_Theta);
        plt.xlabel(r"$\Delta \theta$ / -")
        plt.ylabel(r"$L_\theta(\Delta \theta)$");
        plt.title("Declination angle optimization")
        
        plt.figure();
        plt.polar(Theta_Plot, Radii_Fit, label="Raw data");
        plt.polar(Theta_Plot,R(coeff_R(Theta+ThetaOpt,Radii_Fit),Theta+ThetaOpt), label=r"$\Delta \theta$ optimized");
        plt.title("Declination angle optimization")
        plt.legend();
        
    
    #%% find origin perpendicular to symetry axis:
    Theta_Decl = Theta + ThetaOpt;
    
    dxOpt = sp_opt.minimize_scalar(L_TestX,args=(Theta_Decl,Radii_Fit,Theta_Decl)).x;
    
    Theta_dxOpt = Theta_TestX(dxOpt,Theta_Decl,Radii_Fit);
    Radii_dxOpt = Radii_TestX(dxOpt,Theta_Decl,Radii_Fit);
    coeff_dxOpt = coeff_R(Theta_dxOpt,Radii_dxOpt); 
       
    if debug:
        
        dx_Vector = np.linspace(-50,50,1000);
        L_dx = np.zeros_like(dx_Vector);
        
        it = np.nditer(dx_Vector,flags=['f_index']);
        for dx in it:
            L_dx[it.index] = L_TestX(dx,Theta_Decl,Radii_Fit,Theta_Decl);
            
        
        plt.figure();
        plt.plot(dx_Vector,L_dx);
        plt.xlabel(r"$\Delta x$ / px")
        plt.ylabel(r"$L_x(\Delta x)$");
        plt.title("Centroid optimization perpendicular to symmetry axis")
        
        plt.figure();
        plt.plot(Radii_Fit*np.cos(Theta_Plot),Radii_Fit*np.sin(Theta_Plot),'.',label="Raw Data");
        plt.axis('equal');
        plt.plot(R(coeff_dxOpt,Theta)*np.cos(Theta_Decl[::-1]+np.pi/2)-dxOpt,R(coeff_dxOpt,Theta)*np.sin(Theta_Decl[::-1]+np.pi/2),'.',label=r"$\Delta \theta \, & \, \Delta x$ optimized");
        plt.legend()
        
    #%% find origin along symetry axis:
    Theta_Decl = Theta + ThetaOpt;
        
    dyOpt = sp_opt.minimize_scalar(L_TestY,args=(Theta_dxOpt,Radii_dxOpt,Theta_Decl)).x;
    
    Theta_dyOpt = Theta_TestY(dyOpt,Theta_dxOpt,Radii_dxOpt);
    Radii_dyOpt = Radii_TestY(dyOpt,Theta_dxOpt,Radii_dxOpt);
    coeff_dyOpt = coeff_R(Theta_dyOpt,Radii_dyOpt);
        
    if debug:
        
        dy_Vector = np.linspace(-50,50,1000);
        L_dy = np.zeros_like(dy_Vector);
        
        it = np.nditer(dy_Vector,flags=['f_index']);
        for dy in it:
            L_dy[it.index] = L_TestY(dy,Theta_dxOpt,Radii_dxOpt,Theta_Decl);
            
        
        plt.figure();
        plt.plot(dy_Vector,L_dy);
        plt.xlabel(r"$\Delta y$ / px")
        plt.ylabel(r"$L_y(\Delta y)$");
        plt.title("Centroid optimization along symmetry axis")
        
        plt.figure();
        plt.plot(Radii_Fit*np.cos(Theta_Plot),Radii_Fit*np.sin(Theta_Plot),'.', label="Raw data");
        plt.axis('equal');
        plt.plot(R(coeff_dyOpt,Theta)*np.cos(Theta_Decl[::-1]+np.pi/2)-dxOpt,R(coeff_dyOpt,Theta)*np.sin(Theta_Decl[::-1]+np.pi/2)-dyOpt,'.', label=r"$\Delta \theta \, & \, \Delta x & \, \Delta y$ optimized");
        plt.legend()
        
    return coeff_dyOpt, ThetaOpt, dxOpt, dyOpt;
    
#%% calculate volume:
def integrand(theta,coeff):
    return (R(coeff,theta)**3)*np.sin(theta);  