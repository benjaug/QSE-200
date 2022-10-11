#!/usr/bin/env python
# coding: utf-8

# # The Kronig-Penney Model

# The Kronig-Penney model is a simple model of electrons moving in a 1D potential. It is very reminiscent of the approach we took in [Section 6](test-label), although allows us to look at the problem analytically. The approach is as follows: consider the Schrodinger equation 
# 
# \begin{equation}
# - \frac{\hbar^2}{2m} \frac{d^2 \psi}{dx^2} + V(x) \psi = E \psi 
# \end{equation}
# 
# where $ V(x) $ is the potential due to a series of Delta functions that are evenly spaced and all have the same strength.

# The solutions must satisfy 
# ```{math}
# :label: KP-Econd
# \cos(qa) = \cos(ka) + u \frac{\sin(ka)}{ka}. 
# ```
# The left-hand side is clearly restricted $[-1,1]$ while the right-hand side is not. For the values of $k$ that make the right-hand side have magnitude greater than $1$, there will be no allowed energy eigenstates---there will be gaps in the energy spectrum. On the other hand, when $k$ takes a value such that the right-hand side has magnitude less than $1$, there will be a valid solution and allowed energy eigenstates---this leads to the formation of energy bands over the range of permissible $k$ values.

# In[40]:


import numpy as np
import matplotlib.pyplot as plt


# To investigate this, let's look at the equation graphically. Since the left-hand side of [](KP-Econd) is always in the range $[-1,1]$, we'll draw two horizontal lines at $\pm1$. We can then plot the right-hand side for a few different values of $u$ and look at where this falls in the range that solutions exist.

# In[97]:


q = np.linspace(0.00001,15,300)
RHSlist = np.cos(q)+1.1*np.sin(q)/q
[plt.axvline(x=i, ymin=0.15, ymax=+0.71, alpha=1, lw=2, color="grey") for i in q[(RHSlist>=-1) & (RHSlist<=1)]]
plt.plot(q, RHSlist, 'b', label="u=1.1")
plt.axhline(+1,color="k", linestyle="--")
plt.axhline(-1,color="k", linestyle="--")
plt.ylim(-1.5,2)
plt.xlabel('k or q')
plt.ylabel('Amplitude')
plt.legend();


# In[313]:


q = np.linspace(0.00001,16,300)
RHSlist = np.cos(q)+10*np.sin(q)/q
[plt.axvline(x=i, ymin=0.15, ymax=+0.71, alpha=1, lw=2, color="grey") for i in q[(RHSlist>-1) & (RHSlist<1)]]
plt.plot(q, RHSlist, 'b', label="u=10")
plt.axhline(+1,color="k", linestyle="--")
plt.axhline(-1,color="k", linestyle="--")
plt.ylim(-1.5,2)
plt.xlabel('q')
plt.ylabel('cos(kL)')
plt.legend();


# In[345]:


q = np.linspace(0.00001,16,30000)
k = np.linspace(0.00001,16,30000)
u = 1
RHSlist = np.cos(q)+u*np.sin(q)/q

# find crossing from +1+epsilon to +1-epsilon based on sign change
rsign = np.sign(RHSlist-1)
signchange = ((np.roll(rsign, 1) - rsign) != 0).astype(int)
plus_ones = np.argwhere(signchange)[1:]

# find crossing from -1+epsilon to -1-epsilon based on sign change
rsign = np.sign(RHSlist+1)
signchange = ((np.roll(rsign, 1) - rsign) != 0).astype(int)
minus_ones = np.argwhere(signchange)[1:]
    
# re-order bands to have increasing index number
band_limits = np.concatenate((plus_ones, minus_ones),axis=1)
for i in range(band_limits.shape[1]):
    if band_limits[i,0] > band_limits[i,1]:
        band_limits[i,[0,1]] = band_limits[i,[1,0]]
        
# now go along the q axis and find allowed energies
qsolve = []
ksolve = []
for b in range(3):
    for i in np.arange(band_limits[b,0], band_limits[b,1]):
        ksolve.append(k[i])
        qsolve.append(np.arccos(np.cos(q[i])+u*np.sin(q[i])/q[i]))


# In[346]:


plt.plot(qsolve,ksolve)


# In[ ]:




