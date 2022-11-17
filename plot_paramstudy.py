import numpy as np
import pylab as plt
from model_thermdelay import *

#==============================================================================#
'''
Plotting script to produce Figure 4 in the ApJL,
Salvesen (2022): An electron-scattering time delay in black hole accretion disks
`python plot_paramstudy.py`
'''
#==============================================================================#
# Input parameters for a representative BH XRB in the intermediate state
inc     = 0 * u.deg  # Inclination angle
m       = 10         # BH mass, m = M / M_sun
a       = 0.9        # BH spin
l_d     = 0.2        # Disk luminosity (Eddington-scaled), l = L_disk / L_Edd
alpha   = 0.2        # Effective viscosity parameter
fsz     = 0          # Frac of accretion power dissipated in disk surface layers
beta    = 0.5        # Albedo (reflectance) of the disk surface
zeta    = 1          # Parameter specifying form of hydrostatic equilibrium eqn
xi      = 1          # Parameter specifying form of radiative diffusion equation
l_c     = 0.05       # Corona luminosity (Eddington-scaled), l_c = L_c / L_Edd
h_c     = 10         # Corona height, h_c = H_c / R_g
X       = 0.744      # Hydrogen mass fraction
Y       = 0.242      # Helium mass fraction
Z       = 0.014      # Metals mass fraction
GR_flag = True       # Relativistic corrections turned on (True) or off (False)
rhoStar_rho = 0.1    # Density ratio, effective photosphere / disk mid-plane

# Array of disk radii
def innermost_stable_circular_orbit(a):
    Z1     = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
    Z2     = np.sqrt(3 * a**2 + Z1**2)
    r_isco = 3 + Z2 - np.sign(a) * np.sqrt((3 - Z1) * (3 + Z1 + 2*Z2))
    return r_isco
r_out = 50
Nr    = 1000
r     = np.linspace(innermost_stable_circular_orbit(a=a), r_out, Nr+1)[1:]

# Array of disk azimuthal angles
phi_min = 0 * u.rad
phi_max = 2 * np.pi * u.rad
Nphi    = 100
phi     = np.linspace(phi_min, phi_max, Nphi)

# Alternative parameters
new_m       = 1e9
new_GR_flag = False
new_xi      = 2
new_fsz     = 0.05
new_zeta    = 2
new_l_c     = 0.1

#==============================================================================#

# Instances of the `thermdelay` class
inst_ref  = thermdelay(inc=inc, m=m,     a=a, l_d=l_d, alpha=alpha, fsz=fsz,     beta=beta, zeta=zeta,     xi=xi,     rhoStar_rho=rhoStar_rho, l_c=l_c,     h_c=h_c, X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_m    = thermdelay(inc=inc, m=new_m, a=a, l_d=l_d, alpha=alpha, fsz=fsz,     beta=beta, zeta=zeta,     xi=xi,     rhoStar_rho=rhoStar_rho, l_c=l_c,     h_c=h_c, X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_GR   = thermdelay(inc=inc, m=m,     a=a, l_d=l_d, alpha=alpha, fsz=fsz,     beta=beta, zeta=zeta,     xi=xi,     rhoStar_rho=rhoStar_rho, l_c=l_c,     h_c=h_c, X=X, Y=Y, Z=Z, GR_flag=new_GR_flag)
inst_xi   = thermdelay(inc=inc, m=m,     a=a, l_d=l_d, alpha=alpha, fsz=fsz,     beta=beta, zeta=zeta,     xi=new_xi, rhoStar_rho=rhoStar_rho, l_c=l_c,     h_c=h_c, X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_fsz  = thermdelay(inc=inc, m=m,     a=a, l_d=l_d, alpha=alpha, fsz=new_fsz, beta=beta, zeta=zeta,     xi=xi,     rhoStar_rho=rhoStar_rho, l_c=l_c,     h_c=h_c, X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_zeta = thermdelay(inc=inc, m=m,     a=a, l_d=l_d, alpha=alpha, fsz=fsz,     beta=beta, zeta=new_zeta, xi=xi,     rhoStar_rho=rhoStar_rho, l_c=l_c,     h_c=h_c, X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_lc   = thermdelay(inc=inc, m=m,     a=a, l_d=l_d, alpha=alpha, fsz=fsz,     beta=beta, zeta=zeta,     xi=xi,     rhoStar_rho=rhoStar_rho, l_c=new_l_c, h_c=h_c, X=X, Y=Y, Z=Z, GR_flag=GR_flag)

# Light-travel time delays
t_lt      = inst_ref.light_travel_time_delay(r=r, phi=phi)[:,0]
t_lt_m    = inst_m.light_travel_time_delay(r=r, phi=phi)[:,0]
t_lt_GR   = inst_GR.light_travel_time_delay(r=r, phi=phi)[:,0]
t_lt_xi   = inst_xi.light_travel_time_delay(r=r, phi=phi)[:,0]
t_lt_fsz  = inst_fsz.light_travel_time_delay(r=r, phi=phi)[:,0]
t_lt_zeta = inst_zeta.light_travel_time_delay(r=r, phi=phi)[:,0]
t_lt_lc   = inst_lc.light_travel_time_delay(r=r, phi=phi)[:,0]

# Thermalization time delays
t_th      = inst_ref.thermalization_time_delay(r=r)
t_th_m    = inst_m.thermalization_time_delay(r=r)
t_th_GR   = inst_GR.thermalization_time_delay(r=r)
t_th_xi   = inst_xi.thermalization_time_delay(r=r)
t_th_fsz  = inst_fsz.thermalization_time_delay(r=r)
t_th_zeta = inst_zeta.thermalization_time_delay(r=r)
t_th_lc   = inst_lc.thermalization_time_delay(r=r)

# Peaks
print("")
print(f"Multiplicitive factor by which the peak (t_th / t_lt) changes for a = {a:.3f}:")
print(f"  {np.max(t_th_m    / t_lt_m)    / np.max(t_th / t_lt):.2f} : m = {m:.2f} -> {new_m:.2f}")
print(f"  {np.max(t_th_GR   / t_lt_GR)   / np.max(t_th / t_lt):.2f} : GR = {GR_flag} -> {new_GR_flag}")
print(f"  {np.max(t_th_xi   / t_lt_xi)   / np.max(t_th / t_lt):.2f} : xi = {xi:.2f} -> {new_xi:.2f}")
print(f"  {np.max(t_th_fsz  / t_lt_fsz)  / np.max(t_th / t_lt):.2f} : fsz = {fsz:.2f} -> {new_fsz:.2f}")
print(f"  {np.max(t_th_zeta / t_lt_zeta) / np.max(t_th / t_lt):.2f} : zeta = {zeta:.2f} -> {new_zeta:.2f}")
print(f"  {np.max(t_th_lc   / t_lt_lc)   / np.max(t_th / t_lt):.2f} : l_c = {l_c:.2f} -> {new_l_c:.2f}")
print("")

# Colors
color_ref  = 'black'
color_xi   = 'C0'
color_m    = 'C1'
color_zeta = 'C2'
color_GR   = 'C3'
color_lc   = 'C4'
color_fsz  = 'C5'

#==============================================================================#

fout = 'results/paramstudy.pdf'

# Plotting options:
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathrsfs, anyfontsize}')
xlabel     = r'${\rm Disk\ Radius,}\ r \equiv R / R_{\rm g}$'
ylabel     = r'$t_{\rm th} / t_{\rm lt}$'
xmin, xmax = 0, r_out
ymin, ymax = 0.1, 20000
opts_fig   = {'figsize':(8.4, 5.6), 'dpi':300}
opts_lrbt  = {'left':0.175, 'right':0.975, 'bottom':0.20, 'top':0.97}
opts_xlab  = {'xlabel':xlabel, 'fontsize':28, 'ha':'center', 'va':'top'}
opts_ylab  = {'ylabel':ylabel, 'fontsize':28, 'ha':'center', 'va':'center'}
opts_xtmaj = {'axis':'x', 'direction':'out', 'length':10, 'width':2,
              'labelsize':24, 'pad':5}
opts_xtmin = {'axis':'x', 'direction':'out', 'length':5,  'width':2}
opts_ytmaj = {'axis':'y', 'direction':'out', 'length':10, 'width':2,
              'labelsize':24, 'pad':5}
opts_ytmin = {'axis':'y', 'direction':'out', 'length':5,  'width':2}
opts_plot  = {'linewidth':2, 'zorder':2}

# Setup
fig = plt.figure(**opts_fig)
fig.subplots_adjust(**opts_lrbt)
ax = fig.add_subplot(111)
# Labels
ax.set_xlabel(**opts_xlab)
ax.set_ylabel(**opts_ylab)
# Limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# Scale
ax.set_xscale('linear')
ax.set_yscale('log')
# Ticks
ax.tick_params(which='major', **opts_xtmaj)
ax.tick_params(which='minor', **opts_xtmin)
ax.tick_params(which='major', **opts_ytmaj)
ax.tick_params(which='minor', **opts_ytmin)
# Tick locations
y_tick_locs = [ 0.1,   1,   10,   100,   1000,   10000  ]
y_tick_labs = ['0.1', '1', '10', '100', '1000', '10,000']
ax.set_yticks(y_tick_locs)
ax.set_yticklabels(y_tick_labs, minor=False)

# Plot: t_th / t_lt
ax.plot([xmin, xmax], [1, 1], linewidth=2, linestyle='dotted', color='black', zorder=2)
l_m,    = ax.plot(r, t_th_m    / t_lt_m,    **opts_plot, linestyle='solid',  color=color_m)
l_GR,   = ax.plot(r, t_th_GR   / t_lt_GR,   **opts_plot, linestyle='solid',  color=color_GR)
l_xi,   = ax.plot(r, t_th_xi   / t_lt_xi,   **opts_plot, linestyle='solid',  color=color_xi)
l_fsz,  = ax.plot(r, t_th_fsz  / t_lt_fsz,  **opts_plot, linestyle='solid',  color=color_fsz)
l_zeta, = ax.plot(r, t_th_zeta / t_lt_zeta, **opts_plot, linestyle='solid',  color=color_zeta)
l_lc,   = ax.plot(r, t_th_lc   / t_lt_lc,   **opts_plot, linestyle='solid',  color=color_lc)
l_ref,  = ax.plot(r, t_th      / t_lt,      **opts_plot, linestyle='dashed', color=color_ref)

# Legend
leg = ax.legend(
    [l_ref, l_xi, l_m, l_zeta, l_GR, l_lc, l_fsz],
    ["", "", "", "", "", "", "", ""],
    bbox_to_anchor=(0.575, 0.265), handletextpad=0, labelspacing=0.33)
leg.set_zorder(1)
ax.text(0.975, 0.975,
    r"${\rm Reference}\vspace{5pt}$"
    r"$\\ \begin{array}{rcl}$"
    r"$         \xi & = & "      + f"{new_xi}"              + "$"
    r"$\\         m & = & 10^{"  + f"{np.log10(new_m):.0f}" + "}$"
    r"$\\     \zeta & = & "      + f"{new_zeta}"            + "$"
    r"$\\  {\rm GR} & = & {\rm " + f"{new_GR_flag}"         + "}$"
    r"$\\ l_{\rm c} & = & "      + f"{new_l_c}"             + "$"
    r"$\\         f & = & "      + f"{new_fsz}"             + "$"
    r"$\end{array}$",
    transform=ax.transAxes, ha='right', va='top', zorder=2)

# Plot grid lines
plt.grid(linewidth=1, color='LightGray')
ax.set_axisbelow(True)

# Fix the issue of different fonts
plt.rcParams.update({'font.family':'custom'})
# Save the figure
fig.savefig(fout, bbox_inches=0, dpi=opts_fig['dpi'])
plt.close()

#==============================================================================#
