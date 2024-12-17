import os 
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from astropy.table import QTable, Table, Column, MaskedColumn, join

from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import fnmatch
from os import path, listdir, chdir, mkdir
from astropy.io import fits
from subprocess import call

from scipy.integrate import quad
from scipy.stats import chi2,norm

import glob
from astropy.wcs import WCS
import sys

from fermipy.gtanalysis import GTAnalysis
from fermipy.plotting import ROIPlotter, SEDPlotter

def sumTS(ts1, nnn1, ntrials=20, resample = True, verbose =False): #ONLY used for CFs 

        tsfinal=ts1
        overall_TS_psr=np.zeros([nnn1,ntrials])
        if verbose:
            print('input nnn1 is', nnn1)

        for k in range(ntrials):
            if resample:
                #np.random.choice(np.arange(0,len(tsfinal)), size = nnn1, replace = False)
                indices_psr=np.random.choice(np.arange(0,len(tsfinal)), size = nnn1, replace = False)
                #print(indices_psr)
            else:
                indices_psr=np.arange(nnn1) #no resample just shuffle the input array indicies --> deprecated
                np.random.shuffle(indices_psr)

            TS300sub=np.array([]) #this is the part I don't really follow, why is the variable called this --> for no reason I guess
            for i in range(nnn1): #picking subsample out of large sample  
                TS300sub=np.append(TS300sub,(tsfinal[indices_psr[i]]))

            TS_acc = np.array([])

            for i in range(nnn1):
                TS_acc=np.append(TS_acc, np.sum(TS300sub[:i])) #cumulative sum 

            overall_TS_psr[:,k]=TS_acc #assigning the values of the cumulative sum, TS_acc should have dimensions [nnn1, ntrials] --> has dims [nn1, ] makes sense, repeats k trials
            if verbose:
                print('TS_acc shape is ', TS_acc.shape)
            
        todraw_psr=np.zeros(nnn1)
        todraw_psr_err=np.zeros(nnn1)
        for i in range(nnn1):
            todraw_psr[i]=np.mean(overall_TS_psr[i,:]) #taking mean of each sum step over the trials
            todraw_psr_err[i]=np.std(overall_TS_psr[i,:]) #standard error 
        return(todraw_psr, todraw_psr_err)
    
def resampling_CFTS(targ_ts_dist, cf_ts_dist, ntrials=100):
        overall_tscf=np.zeros([30,ntrials])
        TScf_final=cf_ts_dist
        ncfs = len(targ_ts_dist)
        TScf_final[TScf_final<0]=0
        histbins=np.arange(0,31)
        for i in range(ntrials):
            ind=np.random.choice(np.arange(0,len(TScf_final)), size = ncfs, replace = False)
            tspart=TScf_final[ind]
            overall_tscf[:,i]=np.histogram(tspart,histbins)[0]

        bs_tscf=np.zeros(30)
        bs_tscf_std=np.zeros(30)

        for i in range(30):
            bs_tscf[i]=np.mean(overall_tscf[i,:])
            bs_tscf_std[i]=np.std(overall_tscf[i:])

        return(bs_tscf,bs_tscf_std)   
    
def deltaTSsum(pop_ts, cf_ts, pop_name, cf_name, nos=0, doing=True, n_bs=100, n_samples=1000):
        
        if doing:
            if path.exists(pop_name+'_dTS.npy') & path.exists(pop_name+'_dis.npy')\
                & path.exists(cf_name+'_dTS.npy') & path.exists(cf_name+'_dis.npy'):
                print('Unsorted Stack Already Completed.')
            else:
                TSv2 = pop_ts
                n_source = len(TSv2)
                final_dTS_fs = np.array([])
                final_dis_fs = np.array([])
                for i in range(n_samples):
                    cf2,cf2err=sumTS(TSv2,n_source,n_bs)
                    final_dTS_fs=np.append(final_dTS_fs, cf2[-1])
                    final_dis_fs=np.append(final_dis_fs,cf2err[-1])
                np.save(pop_name+'_dTS.npy',final_dTS_fs,allow_pickle=True)
                np.save(pop_name+'_dis.npy',final_dis_fs,allow_pickle=True)

                TSv2=cf_ts
                final_dTS_fs=np.array([])
                final_dis_fs=np.array([])
                for i in range(n_samples):
                    cf2,cf2err=sumTS(TSv2,n_source,n_bs)
                    final_dTS_fs=np.append(final_dTS_fs, cf2[-1])
                    final_dis_fs=np.append(final_dis_fs,cf2err[-1])
                np.save(cf_name+'_dTS.npy',final_dTS_fs,allow_pickle=True)
                np.save(cf_name+'_dis.npy',final_dis_fs,allow_pickle=True)
        else:
            TSv2 = tsss
            n_source = nos
            final_dTS_fs = np.array([])
            final_dis_fs = np.array([])
            for i in range(n_samples):
                cf2,cf2err=sumTS(TSv2,n_source,n_bs)
                final_dTS_fs=np.append(final_dTS_fs, cf2[-1])
                final_dis_fs=np.append(final_dis_fs,cf2err[-1])
            #print(len(final_dTS_fs),len(final_dis_fs))
            return(final_dTS_fs, final_dis_fs)
        

    
def draw_the_plot(pop_ts, cf_ts, pop_name, cf_name, dof=2):#, version = 'v0'):
        #v0: original;
        #v1: TS stack sorted by RA, error bars on the TS stack;
        #v2: TS stack sorted by RA, error bars triangular;
        #v3: TS stack sorted by RA, fainter blue lines indicating some possible re-ordering. 
        #v4: TS stack average, fainter blue lines indicating some possible re-ordering. 
        fig_label=input("Name of population: ")
        fig1 = plt.figure(constrained_layout=True,figsize=(15,7))
        gs1 = fig1.add_gridspec(2,2)
        
        ###TS distribution of the sources
        f1_ax1=fig1.add_subplot(gs1[0,0])
        pos_ts2=cf_ts
        pos_ts=pop_ts
        #print(pos_ts)
        where_are_NaNs = np.isnan(pos_ts)
        pos_ts[where_are_NaNs] = 0
        #print(pos_ts.max())
        pos_ts[pos_ts<0]=0
        
        f1_ax1.set_yscale('log')
        f1_ax1.set_xlabel('Test Statistic (TS)',fontsize=14)
        f1_ax1.set_ylabel('# '+pop_name,fontsize=14)
        f1_ax1.set_xlim(0,np.amax(pos_ts)+10)
        f1_ax1.set_ylim(0.9,len(pos_ts2))

        bins=np.arange(0, int(np.amax(pos_ts))+2)
        extrabins=np.linspace(0,1,20)
        (nn, nnbins, nnpatch)=f1_ax1.hist([pos_ts],histtype='step', bins=bins,label=[fig_label], color = ['royalblue'])
        newnbins=np.append(extrabins,np.linspace(1,(nnbins.max()-1),(nnbins.max()-1)*4))
        chi2int=np.zeros(len(nnbins))
        for i in range(len(nnbins)-1):
            I=quad(chi2.pdf,nnbins[i],nnbins[i+1],args=(dof,0,1))
            chi2int[i]=len(pos_ts)*I[0]/2
        f1_ax1.plot(nnbins, chi2int*2,drawstyle='steps-post',label=r"${\chi}^2/2$, dof="+str(dof),c='orange')
        f1_ax1.legend(loc='upper right',fontsize=14)

        ###Sampled TS distribution of the controls 
        f1_ax2=fig1.add_subplot(gs1[1,0])
        f1_ax2.axvline(25,c='r',ls='--',label="TS = 25")
        f1_ax2.set_yscale('log')
        f1_ax2.set_xlabel('Test Statistic (TS)',fontsize=14)
        f1_ax2.set_ylabel('# Test Sources',fontsize=14)
        f1_ax2.set_xlim(0,np.amax(pos_ts)+10)
        f1_ax2.set_ylim(0.9,len(pos_ts2))
        
        bins=np.arange(0, int(np.amax(pos_ts2))+2)
        extrabins=np.linspace(0,1,20)
        bs_tscf, bs_tscf_std=resampling_CFTS(pos_ts, pos_ts2, ntrials=1000)
        f1_ax2.plot(np.arange(-1,30),np.append([0],bs_tscf),drawstyle='steps-post',label='Test Sources',c='seagreen',lw=0.8)
        f1_ax2.errorbar(np.arange(0,30)+0.5,bs_tscf,yerr=bs_tscf_std,c='seagreen',ls='none',lw=0.8)

        newnbins=np.append(extrabins,np.linspace(1,(nnbins.max()-1),(nnbins.max()-1)*4))
        chi2int=np.zeros(len(nnbins))
        for i in range(len(nnbins)-1):
            I=quad(chi2.pdf,nnbins[i],nnbins[i+1],args=(dof,0,1))
            chi2int[i]=len(pos_ts)*I[0]/2
        f1_ax2.plot(nnbins, chi2int*2,drawstyle='steps-post',label=r"${\chi}^2/2$, dof="+str(dof),c='orange')

        f1_ax2.legend(loc='upper right',fontsize=14)

        final_dTS_fs3=np.load(pop_name+'_dTS.npy',allow_pickle=True)
        final_cTS=np.load(cf_name+'_dTS.npy',allow_pickle=True)
        final_cTS_dis=np.load(cf_name+'_dis.npy',allow_pickle=True)
        final_dis_fs3=np.load(pop_name+'_dis.npy',allow_pickle=True)

        ###Cumulative TS distribution
        f1_ax3=fig1.add_subplot(gs1[:,1])
        #pos_ts=np.load(file1, allow_pickle=True)
        #pos_ts2=np.load(file2, allow_pickle=True)
        nnn1=len(pos_ts)
        
        todraw_psr1 = np.cumsum(pos_ts)
        resampling_size=1000
        to_be_averaged = np.zeros((nnn1, resampling_size))
        for ijk in range(resampling_size):
            np.random.shuffle(pos_ts)
            todraw_psr = np.cumsum(pos_ts)
            f1_ax3.plot(np.arange(nnn1), todraw_psr, c='lightskyblue', markersize = 2, zorder = 1, alpha=0.1, ls='-')
         
        f1_ax3.plot(np.arange(nnn1), todraw_psr1, c='b', ls='-', label=fig_label)
        
        todraw_cfs, todraw_cfs_err=sumTS(pos_ts2, nnn1, 1000)
        #print(todraw_cfs[-1],norm.fit(final_cTS)[0])
        #print(todraw_cfs, todraw_cfs_err)
        #print(norm.fit(final_cTS)[0], norm.fit(final_cTS_dis)[0]) 
        while (abs(todraw_cfs[-1]-norm.fit(final_cTS)[0])>5) or abs((todraw_cfs_err[-1]-norm.fit(final_cTS_dis)[0])>50):
            todraw_cfs, todraw_cfs_err=sumTS(pos_ts2, nnn1, 1000)
        #print(todraw_cfs, todraw_cfs_err)
        print(todraw_psr[-1], todraw_cfs[-1])
        print(todraw_psr[-1]-todraw_cfs[-1])
     
        
        f1_ax3.plot(np.arange(nnn1), todraw_cfs, c='g', ls='-',  ms=2, label='Test Sources')
        f1_ax3.fill_between(np.arange(nnn1), todraw_cfs+todraw_cfs_err, todraw_cfs-todraw_cfs_err, zorder = 2, alpha=0.3,interpolate=False, step=None, color='lawngreen')
        

        (nn,nnbins,nnpatch) = plt.hist([pos_ts], bins, histtype='step', color = ['b'], alpha  = 0)
        
        
        bins=603
        asum = np.zeros(bins)
        for j in range(bins): #chi^2 integration for comparison 
            (nn,nnbins,nnpatch)= plt.hist([pos_ts], j+1, histtype='step', color = ['b'], alpha = 0)
            chi2int2=np.zeros(len(nnbins))
            for i in range(len(nnbins)-1):
                I=quad(chi2.pdf,nnbins[i],nnbins[i+1],args=(2,0,1))
                chi2int2[i]=(j+1)*I[0]/2
                #print(chi2int2)
            asum[j] = np.sum(np.flip(chi2int2))
        
        f1_ax3.plot(asum, label = r'$\chi^{2}/2$ sum', color = 'orange')
        f1_ax3.set_xlim(1,nnn1+1)
        f1_ax3.set_ylim(1, 1.2*(todraw_psr[-1]))
        f1_ax3.set_xlabel('Number in Stack',fontsize=14)
        f1_ax3.set_ylabel(ylabel=r'$\Sigma$TS',fontsize=14)
        f1_ax3.legend(loc='upper left', fontsize=14)
       

        fig1.savefig(pop_name+'overall.png', transparent = False, facecolor = 'w')
        return
        

    


