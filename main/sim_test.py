from wavevector import Wavevector
from wavefunction import Wavefunction
import numpy as np
from matplotlib import pyplot as plt

L,C,ħ,Φo = 10e-6,10e-15,1.05e-34,2.07e-15
#L,C,ħ,Φo = 1,1,1,1  # Φo is flux quantum

def V(Φ):
    #return Φ*0
    return Φ**2/(12*L)

x_range = 3*Φo
dim_info = ((-x_range, x_range, 41),)
masses = (C,)
σ = np.sqrt(ħ/2*np.sqrt(L/C))
ω = 1/np.sqrt(L*C)
wv_o = Wavevector.from_wf(Wavefunction.init_gaussian((0.5*Φo, σ)), *dim_info)
r = wv_o.evolve(V, masses, (0, 2*np.pi/ω*1e-7), frames = 3, t_dep = False)
plt.plot(np.abs(r.y[:,2])**2)
plt.plot(np.abs(r.y[:,1])**2)

plt.plot(np.abs(r.y[:,0])**2)
plt.show()



