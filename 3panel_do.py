import numpy as np 
from sigmaTS_plots import *

TSin = 'GC_TSs-Copy1.npy'
TScfin = 'TSsCFs-Copy1.npy'

#TSin = input('Path to your target TS dirstirbution:')
#TScfin = input('Path to your CF TS dirstirbution:')

TS = np.load(TSin, allow_pickle = True)
TScf = np.load(TScfin, allow_pickle = True)

pop_name = input('Target population name: ')
cf_name = input('CF population name: ')

name1 = pop_name
name2 = cf_name

cf_ts_dist= TScf
pop_ts_dist = TS

dof=1

deltaTSsum(pop_ts = pop_ts_dist, cf_ts = cf_ts_dist, pop_name = name1, cf_name = name2)

final_dTS_fs3=np.load(name1+'_dTS.npy',allow_pickle=True)
final_cTS=np.load(name2+'_dTS.npy',allow_pickle=True)
final_cTS_dis=np.load(name2+'_dis.npy',allow_pickle=True)
final_dis_fs3=np.load(name1+'_dis.npy',allow_pickle=True)



draw_the_plot(pop_ts=pop_ts_dist, cf_ts=cf_ts_dist, pop_name = name1, cf_name = name2)