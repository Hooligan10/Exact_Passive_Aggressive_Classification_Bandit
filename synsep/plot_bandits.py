from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
import os


with open('abc.txt', 'r') as f:
	x = f.read().splitlines()
	aa = (eval(x[0]))


filename = 'xyz'

plt.clf()
plt.grid()
	
erph = aa['error_rate_ph']
erb = aa['error_rate_b']
er = aa['error_rate_epabf']
er1 = aa['error_rate_epabf1']
er2 = aa['error_rate_epabf2']

rounds = aa['rounds']
rrph = rrb = rr = rr1 = rr2 = rrpa = rrpa1 = rrpa2 = rrc = rounds

plt.plot(rrph, erph, 'o--', label='BPA', color='brown',linewidth = 1,markersize = 9, markevery = 100)
plt.plot(rrb, erb, 'v--', label='Banditron', color='black',linewidth = 1,markersize = 9, markevery = 100)
plt.plot(rr, er, '*--', label='EPABF', color='green',linewidth = 1,markersize = 9, markevery = 100)
plt.plot(rr1, er1, '^--', label='EPABF-I', color='blue',linewidth = 1,markersize = 9, markevery = 100)
plt.plot(rr2, er2, 'x--', label='EPABF-II', color='red',linewidth = 1,markersize = 9, markevery = 100)

plt.xlabel('Rounds',fontsize = "xx-large")
plt.ylabel('Error Rate',fontsize = "xx-large")
plt.title('SynSep: Error Rate v/s Rounds',fontsize = "xx-large")
plt.legend(loc='upper right',fontsize="large")

savepath = os.path.join(filename + ".png")
	
plt.savefig(savepath)