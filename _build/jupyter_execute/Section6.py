#!/usr/bin/env python
# coding: utf-8

# (test-label)=
# # An Intuitive Picture of Band Structure

# I am following "Calculation of band structures by a discrete variable representation based on Bloch functions" (View online: https://doi.org/10.1119/1.1994858) to make a DVR method for periodic potentials.

# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# ### Using the DVR scheme with periodic boundary conditions

# In[583]:


hbar = 1
mu = 1

# Set these
M=100 # Basis size will be N=2M+1
P=5 # Periodicity

# Determined
xs = np.arange(0, P, P/(2*M+1))


# In[584]:


# Build the Hamiltonian
def make_H(kappa, P, M, Vfunc):
    N = 2*M+1
    Hmat = np.zeros((N,N), dtype=complex)
    Delta = P/N
    Delta_prime = np.pi/N
    omega = 2*np.pi/P
    for i in range(N):
        for j in range(N):
            if i==j:
                Hmat[i,j] = (hbar**2/(2*mu)) * (kappa**2 + omega**2 * (M*(M+1))/3) + Vfunc(i*Delta)
            else:
                factor1 = (-1j*kappa*omega)/(np.sin((i-j)*Delta_prime))
                factor2 = omega**2 * np.cos((i-j)*Delta_prime)/(2*np.sin((i-j)*Delta_prime)**2)
                Hmat[i,j] = (hbar**2/(2*mu)) * (-1)**(i-j) * np.exp(1j*kappa*(i-j)*Delta) * (factor1+factor2)
            
    return Hmat


# In[585]:


def Vfunc(x):
    return 1.5 + 1.5*np.cos(2*np.pi*x/P)


# In[586]:


H=make_H(0.1, P, M, Vfunc)
vals, vecs = np.linalg.eigh(H)


# In[587]:


kappa_vals= np.linspace(-np.pi/P,np.pi/P,100)
bands = np.zeros((6,len(kappa_vals)))
for (nk,kappa) in enumerate(kappa_vals):
    Hmat = make_H(kappa, P, M, Vfunc)
    vals, vecs = np.linalg.eigh(Hmat)
    bands[:,nk] = vals[0:6]


# In[588]:


for band_val in range(6):
    plt.plot(kappa_vals, bands[band_val,:])


# In[589]:


vals, vecs = np.linalg.eigh(make_H(0.2, P, M, Vfunc))
for i in range(3):
    plt.plot(np.abs(vecs[:,i])/np.max(np.abs(vecs[:,i])));


# Now let's try more interesting unit cells. Start with the Kronig-Penney model.

# In[590]:


def Vfunc(x):
    return 3*((np.abs(x) < P/4)|(np.abs(x)>3*P/4))

plt.plot(xs,Vfunc(xs))


# In[ ]:


kappa_vals= np.linspace(-np.pi/P,np.pi/P,100)
bands = np.zeros((6,len(kappa_vals)))
for (nk,kappa) in enumerate(kappa_vals):
    Hmat = make_H(kappa, P, M, Vfunc)
    vals, vecs = np.linalg.eigh(Hmat)
    bands[:,nk] = vals[0:6]


# In[ ]:


for band_val in range(6):
    plt.plot(kappa_vals, bands[band_val,:])


# Also plot the wavefunctions

# In[297]:


vals, vecs = np.linalg.eigh(make_H(0.2, P, M, Vfunc))
for i in range(3):
    plt.plot(np.abs(vecs[:,i])/np.max(np.abs(vecs[:,i])));


# Try the double-well potential that Federico mentioned.

# In[223]:


def Vfunc_sym(x):
    return -1+ 1*((np.abs(x) < P/8)|(np.abs(x)>3.5*P/8)) + 1*((np.abs(x) < 4.5*P/8)|(np.abs(x)>7*P/8))

def Vfunc_asym(x):
    return -1+ 1*((np.abs(x) < 2.5*P/8)|(np.abs(x)>3.5*P/8)) + 1*((np.abs(x) < 4.5*P/8)|(np.abs(x)>7*P/8))

plt.plot(xs,Vfunc_sym(xs))
plt.plot(xs,Vfunc_asym(xs))


# In[224]:


kappa_vals= np.linspace(0,np.pi/P,25)
bands_sym = np.zeros((6,len(kappa_vals)))
for (nk,kappa) in enumerate(kappa_vals):
    Hmat = make_H(kappa, P, M, Vfunc_sym)
    vals, vecs = np.linalg.eigh(Hmat)
    bands_sym[:,nk] = vals[0:6]
    
kappa_vals= np.linspace(0,np.pi/P,25)
bands_asym = np.zeros((6,len(kappa_vals)))
for (nk,kappa) in enumerate(kappa_vals):
    Hmat = make_H(kappa, P, M, Vfunc_asym)
    vals, vecs = np.linalg.eigh(Hmat)
    bands_asym[:,nk] = vals[0:6]


# In[225]:


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for band_val in range(6):
    plt.plot(kappa_vals, bands_sym[band_val,:], color=colors[band_val])
    plt.plot(kappa_vals, bands_asym[band_val,:],'--', color=colors[band_val])
    plt.gca().set_prop_cycle(None)
plt.ylim(0,3)
plt.xlabel("ka")
plt.ylabel("Energy")
plt.title("Solid = symmetric, dashed = asymmetric")


# Also plot the wavefunctions for a particular kappa value

# In[256]:


vals, vecs = np.linalg.eigh(make_H(0.352, P, M, Vfunc_sym))


# In[328]:


plt.plot(np.abs(vecs[:,2])**2)


# And the "soft Coulomb" potential

# In[302]:


def V_C(x):
    return -1**2/np.sqrt(1**2+(x-10/2)**2)


# In[303]:


xs = np.linspace(0,10,100)
plt.plot(xs,V_C(xs))


# In[307]:


P = 10
M = 50
kappa_vals= np.linspace(0,np.pi/P,25)
bands = np.zeros((6,len(kappa_vals)))
for (nk,kappa) in enumerate(kappa_vals):
    Hmat = make_H(kappa, P, M, V_C)
    vals, vecs = np.linalg.eigh(Hmat)
    bands[:,nk] = vals[0:6]


# In[308]:


for band_val in range(6):
    plt.plot(kappa_vals, bands[band_val,:])


# In[313]:


bands[0:4,[0,-1]]


# In[314]:


vals, vecs = np.linalg.eigh(make_H(0.157, P, M, V_C))


# In[326]:


for i in range(4):
    plt.plot(vals[i]+0.3*np.abs(vecs[:,i])**2/np.max(np.abs(vecs[:,i])**2))


# ### Also try the sinc-basis DVR code and program some "periodic" potentials by hand

# In[614]:


hbar = 1
m = 1


# In[615]:


# Function to make the kinetic energy operator
def make_T(x):
    Delta_x = x[1]-x[0]
    N = x.shape[0]
    Tmat = np.zeros((N,N))
    
    # now loop over kinetic energy matrix and fill in matrix elements
    for i in range(N):
        for j in range(N):
            if i==j:
                Tmat[i,j] = (hbar**2/(2*m*Delta_x**2)) * (np.pi**2)/3
            else:
                Tmat[i,j] = (hbar**2/(2*m*Delta_x**2)) * (-1)**(i-j) * 2/(i-j)**2
                
    return Tmat
  
# Function to make the potential energy operator
def make_V(x,Vfunc):
    Vmat = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        Vmat[i,i] = Vfunc(x[i])
    return Vmat

# Function to make the full Hamiltonian
def make_H(x,Vfunc):
    return make_T(x) + make_V(x,Vfunc)


# In[616]:


N = 900
xs = np.linspace(-4,4,N)


# In[661]:


def V_wells(x):
    out = 0
    if np.abs(x) > 3.0:
        out += 200
    else:
        for n in range(-10,10):
            if ((x-0.035) < 0.5*n)&((x+0.035)>0.5*n):
                out += 200
    return out


# In[662]:


plt.plot(xs,[V_wells(xs[i]) for i in range(len(xs))])


# In[663]:


Ham=make_H(xs,V_wells)


# In[664]:


vals, vecs = np.linalg.eigh(Ham)


# In[665]:


plt.plot(vals[0:30],'o')


# In[667]:


for i in range(100):
    plt.axhline(vals[i])
    
plt.plot(xs,[V_wells(xs[i]) for i in range(len(xs))],'k')
plt.ylim(0,300)


# ### Use transfer matrix approach

# In[6]:


m =1


# In[7]:


def DMat(k1, k2):
    res = np.zeros((2,2),dtype=np.complex_)
    res[0,0] = (1 + k2/k1)/2
    res[0,1] = (1 - k2/k1)/2
    res[1,0] = res[0,1]
    res[1,1] = res[0,0]
    return res

def PMat(k, L):
    res = np.zeros((2,2),dtype=np.complex_)
    res[0,0] = np.exp(-1j * k * L)
    res[1,1] = np.exp(1j * k * L)
    return res


# In[103]:


Es = np.arange(0.01, 4.0, 0.0005) #1.6
width_barrier = 0.12
width_gap = 5
Ttrans = np.zeros(Es.size)
i = 0
for E in Es:
    klow = np.emath.sqrt(2 * m * E)
    khigh = np.emath.sqrt(2 * m * (E - Vb))
    res_mat = DMat(klow, khigh) @ PMat(khigh, width_barrier) @ DMat(khigh, klow) @ PMat(klow, width_gap)
    U, V = np.linalg.eig(res_mat)
    diag_res_mat = np.diag([U[0],U[1]])
    res_mat = np.linalg.matrix_power(diag_res_mat,20)
    res_mat = V @ res_mat @ np.linalg.inv(V)
    Ttrans[i] = 1 - np.abs(res_mat[1, 0])**2 / np.abs(res_mat[0,0])**2
    i = i + 1

plt.plot(Es, Ttrans)


# In[ ]:




