import numpy as np;
import eml_toolbox as eml;
from scipy import integrate as sp_int;
import matplotlib.pyplot as plt;
import time;
import os;

#%% configure:

# in case of calibration spheres:
# ThetaOptLim_Rad = 0;
# ThetaOptLim_Rad_Offset = np.deg2rad(0)

# for real samples:
ThetaOptLim_Rad = 1/2*np.pi;
ThetaOptLim_Rad_Offset = np.deg2rad(10)
ThetaOptLim_Rad = (-ThetaOptLim_Rad,+ThetaOptLim_Rad) - ThetaOptLim_Rad_Offset;
ThetaOptLim_Deg = np.rad2deg(ThetaOptLim_Rad);

FilePath = "demo_data/TU Graz side view (density)_radii_edge_cubic_1.0deg.txt"
ImgNr = 1;

if ImgNr > 0:
    # debug yes/no:
    Debug = True;
else:
    Debug = False;

# close old plot windows:
plt.close('all');
    
#%% load data:    
FileName = os.path.basename(FilePath);

# get angle partition:
ThetaStepSizeDeg = float(str.split(str.split(FilePath,'deg')[0],'_')[-1]);

# load data:
with open(FilePath) as f:
    lines = (line for line in f if not line.startswith('#'))
    Data = np.loadtxt(lines, skiprows=1);

# measure performance:
start = time.clock();

# generate default angle vector for raw data:
Data_Theta = np.linspace(0,2*np.pi,int((360/ThetaStepSizeDeg)+1));

#
if ImgNr > 0: 
    Data_Radii = Data[ImgNr-1,3::];
    Data_Radii = Data_Radii.reshape(1,Data_Radii.size);
else:
    Data_Radii = Data[::,3::];
    
# remove lines with NaN:
# affected rows:
NaN_rows = np.isnan(np.sum(Data_Radii,1));
Data_Radii = Data_Radii[~NaN_rows];
    
DataRows = Data_Radii.shape[0];
ParamsOpt = np.zeros((DataRows,10));
V = np.zeros((DataRows,1));

# iterate over all lines: 
for i in range(0,DataRows):
    
    # fit radii data with linear combination of legendre polynomials:
    coeff_opt, ThetaOpt, dxOpt, dyOpt = eml.ef.FitRadii(Data_Theta,Data_Radii[i,::], 
                                                        ThetaOptLim_Rad = ThetaOptLim_Rad, 
                                                        debug = Debug);
    
    # calculate volume:
    ParamsOpt[i,0:7] = coeff_opt;
    ParamsOpt[i,7::] = (ThetaOpt, dxOpt, dyOpt);
    V[i,0] = 2*np.pi/3*sp_int.quad(eml.ef.integrand,0,np.pi,args=(coeff_opt))[0];
    
    print("\rProgress: {:3.0f}%".format(np.round(100*(i+1)/DataRows)),end="");
    
# add newline:
print('');

#%% measure performance:
stop = time.clock();
print("%.2fs" % (stop-start));

#%% show mean volume:
if not Debug:
    print("Mean Volume: %.9f px^3" % np.mean(V));
    
#%% visually check results:
if not Debug:
    plt.figure(figsize=(12, 9), dpi=150);
    plt.subplot(4,2,1)
    plt.plot(V,'.-')
    plt.ylabel('V / px^3')
    plt.xlabel("frame no.")
    plt.title("Volume");
        
    #plt.figure();
    plt.subplot(4,2,3)
    plt.plot(V/np.mean(V)*100,'.-')
    plt.ylim((95.0,105.0)); 
    plt.ylabel(r'V / $\sum_i^N V_i$ / %')
    plt.xlabel("frame no.")
    plt.title(r"relative Volume (V / $\sum_i^N V_i$)");
    
    plt.subplot(4,2,5)
    #plt.figure();
    plt.plot((np.cumsum(V)/np.arange(1,DataRows+1))/np.mean(V)*100,'.-');
    plt.ylim((99.5,100.5));
    plt.ylabel(r'$\sum_i^n V_i$ / $\sum_i^N V_i$ / %')
    plt.xlabel("frame no.")
    plt.title(r"running relative Volume ($\sum_i^n V_i$ / $\sum_i^N V_i$)");
    
    plt.subplot(4,2,2)
    #plt.figure();
    plt.plot(ParamsOpt[:,7]*180/np.pi,'.-');
    plt.ylim(ThetaOptLim_Deg);
    plt.ylabel(r' $\Delta \theta$ / °')
    plt.xlabel("frame no.")
    plt.title("Declination angle")
    
    #plt.figure();
    plt.subplot(4,2,4)
    plt.plot(ParamsOpt[:,8],'.-');
    plt.ylabel('dx / px')
    plt.xlabel("frame no.")
    plt.title("Centroid optimization perpendicular to symmetry axis")
    
    #plt.figure();
    plt.subplot(4,2,6)
    plt.plot(ParamsOpt[:,9],'.-');
    plt.ylabel('dy / px')
    plt.xlabel("frame no.")
    plt.title("Centroid optimization along symmetry axis")
    
    plt.subplot(4,2,7)
    plt.plot(np.amax(Data_Radii,axis=1)/np.sqrt(Data[~NaN_rows,2]/np.pi),'.-');
    plt.ylabel('Ampl. (M1) / -')
    plt.xlabel("frame no.")
    plt.title(r"Amplitude Method 1: max($R(\theta)$) / $\sqrt{\mathrm{pixel\_count}}$")
    
    plt.subplot(4,2,8)
    plt.plot(np.amax(Data_Radii,axis=1)/np.amin(Data_Radii,axis=1),'.-');
    plt.ylabel('Ampl. (M2) / -')
    plt.xlabel("frame no.")
    plt.title(r"Amplitude Method 2: max($R( \theta )$) / min($R( \theta )$)")
    
    plt.tight_layout()


#%% show plots (if not run within ipython console)
plt.show()
