# Parmeterization: Getting simulation running on bokeh server

Mon 05 Jul 2021 06:03:21 PM EDT

Trying to debug parametrization.org

The other day at the meeting we found some problems with the
timescales involved.  Specifically the "mass" (capacitance) has to be
insanely small otherwise the simulation timescale becomes very short.

## Separate Directories

First thing I did was separate the directories again, so that now I
have "param.py" in a project directory.

I then had to add "main" to the path using the "sys" library, which
seems to have worked well.

![Screenshot](figures/Screenshot from 2021-07-05 18-55-13.png)

It is worth plotting the Josephson inductance vs. the critical current to get a sense from about
10 nA to about 1 μA, as these are the scales used in Qubits, and correspond to fluxes on the 
order of Φₒ.

Looking at this, it seems that 1e-6 uA corresponds to about 3 nH.

if f = 1 GHz, then ω ≈ 6 Grad/s so T = 1 ns = 1 / √(L C).
⇒ L C = 1 / T² ⇒ C = 1/(T² L) = 1/(1e-18*1e-9)=1e-27.

Now Δx ~ √(ħ/2 * √(L/C)) ~ √(5e-35*√(1e-9/1e-27)) = √(5e-35*1e9) ~ 2e-13.

made a qubit circuit calculator in desmos

<https://www.desmos.com/calculator/ohpq1wiko6>

The end result was L = 1e-9, C = 1e-15, width is 3 e-16 and period is 6e-12

I feel like I should test it with known quantities, namely set C = ħ² and then
the timesteps shoudl be O(1)

## Tue 06 Jul 2021 09:11:46 AM EDT

OK, I had a few struggles, but it turned out the problem was fundamentally
that the plots weren't updating.  Evolve returns a solve_ivp object, but
I had been assuming it returned a wavevector (i.e. updated in place).

Indeed, that would be more useufl in the long run.

Once I fixed that, I was able to get everything to evolve in place and
look nice on the web page.  I now just need to change it back so
sliders work etc., as I had tod eisable all that during debugging
process.

![screenrecording](figures/Peek 2021-07-06 09-05.gif)
The link above is to a gif, and should work if you go to the file to look at it.
## Aut-scaling of PDF

If you have an x range of X and a stdev of σ, assuming the distribution
is rectangular, the height should be 1/σ

I set it up to calculate the initial σ and then evolve from there.
I'm not sure I can reset x_range internally, but probably it is possible.

## Observing oscillation

I'm going to go back to wavevector to see if I can observe basic oscillations
in matplotlib

If we think about the schrodinger equation

ⅉ ħ ∂ψ/∂t = -ħ²/(2 m) ∂²ψ/∂x² + V(x)

if we divide by  ħ and set m = ħ, we can rewrite this as


ⅉ ∂ψ/∂t = -ħ/(2 m) ∂²ψ/∂x² + V(x)/ħ

thus if we want everything to be O(1), we need V(x) ~ ħ

I tested this and it works fine, as long as we set V = x**2 * ħ

But it doesn't seem to be working ok with the realistic L and C
values.

One quick way to rescale everything in the simulation to match matplotlib
test with wavevector.py is to set L_scale = 0.1, C_scale = 0.1*ħ, and Φₒ = 1.

ⅉ ∂ψ/∂t = -ħ/(2 ħ) ∂²ψ/∂x² + V(x)/ħ

let's take some other scales like σ ~ √(ħ*√(L/C)) ~ √√ħ thus ∂x² ~ √ħ and
∂t ~ 1

ⅉ ∂ψ = -1/(2 m) ∂²ψ/√ħ + V(x)/ħ

# Cleaning up Evolution
Wed 07 Jul 2021 07:15:46 PM EDT

I now have it evolving in place on the website, everything looking smooth and
everything.

I have experience in the past working with solve_ivp where it had real struggles in dealing with small numbers.

Let's try setting L_scale = 0.1*ħ, C_scale = 0.1*ħ.  In that case the
ground state will be width ~ √ħ which we don't want.  We want the width
to be O(1).  So √(L/C) ~ O(1/ħ) ⇒ L/C ~ O(1/ħ²).  But C ~ O(ħ) ⇒ L ~ O(1/ħ).

⇒ L_scale ~ 0.1/ħ

Wow that worked!  Very happy.  There is some unexplained breathing, not sure
what it's due to.  Need to experiment a bit with that.

# Reduce breathing of eigenstate

Try to see if I can reduce the extent to which I see breathing in the eigenstate.

First, just try increasing number of points. 

Didn't help.

I got a bit distracted and decided to add buttons to speed up and slow down the simulation.

Those look nice and work well.  However, I still would like to improve.
# Fix classical evolution of eigenstate

Tue Jul 13 10:18:56 EDT 2021
This definitely gives me an idea for a "find the eigenstates" sandbox, where you try to find
the ground state eigenstates by playing with some knobs on the w.f.
Tue Jul 13 10:18:56 EDT 2021

The classical evolution of the eigenstate should follow the expectation
value of position.  I was trying to use operators etc. to calculate this, but
it wasn't working.  I decided to kludge it using list comprehensions.

# Fix ground state calculation

Thu Jul 15 09:37:51 EDT 2021

From (these notes)[https://www.reed.edu/physics/courses/P342.S10/Physics342/page1/files/Lecture.9.pdf]
equation 9.7

normalized ground state is

(m ω / π ħ )^(1/4) exp(- m ω / 2 ħ x²)

The square of this expression is

√(m ω / π ħ ) exp(- m ω / ħ x²)

How wavefunction from_gaussian function works is it assumes σ is given for the pdf

>>>>>>> refs/remotes/origin/main
return_val *= np.exp(-(x - Xo)**2/(4*σ**2))/(2*π*σ**2)**0.25

In that case, the value we want to use in the init_gaussian call is the one for the pdf, i.e.

- m ω / ħ = - 1 / 2 σ²
⇒ m ω / ħ =  1 / 2 σ²
⇒ ħ / (m ω) =  2 σ²
⇒ ħ / (2 m ω) =  σ²
⇒ √(ħ / (2 m ω)) =  σ
but
m = C
ω = 1/√(L C)

∴ √(ħ √(L C) / (2 C)) =  σ
⇒ √(ħ √(L) / (2 √C)) =  σ
It is convenient to remember Zc = √(L/C) thus
⇒ σ = √(ħ Zc / 2) which I believe is what we were using

-----

ⅉ ħ ∂ψ/∂t = -ħ²/(2 m) ∂²ψ/∂x² + V(x)
∂t ~ 1
∂x ~ 1
ⅉ ħ ∂ψ = -ħ²/(2 m) ∂²ψ + V(x)
ⅉ ∂ψ = -ħ/(2 m) ∂²ψ + V(x)/ħ
C = m ~ ħ
⇒ ⅉ ∂ψ = - ∂²ψ + V(x)/ħ
if ∂t ~ 1, T ~ 1
T = 1/√(LC) = 1/√(L ħ) ⇒ L ~ 1/ħ
and also
V(x) ~ ħ
