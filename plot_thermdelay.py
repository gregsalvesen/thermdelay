import numpy as np
import pylab as plt
from model_thermdelay import *

#==============================================================================#
'''
Plotting script to produce Figures 2 and 3 in the ApJL,
Salvesen (2022): An electron-scattering time delay in black hole accretion disks
`python plot_thermdelay.py`
'''
#==============================================================================#
# Input parameters for a representative BH XRB in the intermediate state
inc     = 0 * u.deg  # Inclination angle
m       = 10         # BH mass, m = M / M_sun
a0      = 0          # Zero BH spin
a9      = 0.9        # High BH spin
a1      = 0.998      # Max  BH spin
l_d     = 0.2        # Disk luminosity (Eddington-scaled), l = L_disk / L_Edd
l_d_lhs = 0.02       # Disk luminosity (Eddington-scaled), low/hard state
alpha   = 0.2        # Effective viscosity parameter
fsz     = 0          # Frac of accretion power dissipated in disk surface layers
beta    = 0.5        # Albedo (reflectance) of the disk surface
zeta    = 1          # Parameter specifying form of hydrostatic equilibrium eqn
xi      = 1          # Parameter specifying form of radiative diffusion equation
l_c     = 0.05       # Corona luminosity (Eddington-scaled), l_c = L_c / L_Edd
h_c1    = 1          # Corona height, h_c = H_c / R_g = 1
h_c10   = 10         # Corona height, h_c = H_c / R_g = 10
h_c100  = 100        # Corona height, h_c = H_c / R_g = 100
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
r0    = np.linspace(innermost_stable_circular_orbit(a=a0), r_out, Nr+1)[1:]
r9    = np.linspace(innermost_stable_circular_orbit(a=a9), r_out, Nr+1)[1:]
r1    = np.linspace(innermost_stable_circular_orbit(a=a1), r_out, Nr+1)[1:]

# Array of disk azimuthal angles
phi_min = 0 * u.rad
phi_max = 2 * np.pi * u.rad
Nphi    = 100
phi     = np.linspace(phi_min, phi_max, Nphi)

#==============================================================================#

# Instances of the `thermdelay` class
inst_a0     = thermdelay(inc=inc, m=m, a=a0, l_d=l_d,     alpha=alpha, fsz=fsz, beta=beta, zeta=zeta, xi=xi, rhoStar_rho=rhoStar_rho, l_c=l_c, h_c=h_c10,  X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_a9     = thermdelay(inc=inc, m=m, a=a9, l_d=l_d,     alpha=alpha, fsz=fsz, beta=beta, zeta=zeta, xi=xi, rhoStar_rho=rhoStar_rho, l_c=l_c, h_c=h_c10,  X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_a1     = thermdelay(inc=inc, m=m, a=a1, l_d=l_d,     alpha=alpha, fsz=fsz, beta=beta, zeta=zeta, xi=xi, rhoStar_rho=rhoStar_rho, l_c=l_c, h_c=h_c10,  X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_h1     = thermdelay(inc=inc, m=m, a=a1, l_d=l_d,     alpha=alpha, fsz=fsz, beta=beta, zeta=zeta, xi=xi, rhoStar_rho=rhoStar_rho, l_c=l_c, h_c=h_c1,   X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_h10    = thermdelay(inc=inc, m=m, a=a1, l_d=l_d,     alpha=alpha, fsz=fsz, beta=beta, zeta=zeta, xi=xi, rhoStar_rho=rhoStar_rho, l_c=l_c, h_c=h_c10,  X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_h100   = thermdelay(inc=inc, m=m, a=a1, l_d=l_d,     alpha=alpha, fsz=fsz, beta=beta, zeta=zeta, xi=xi, rhoStar_rho=rhoStar_rho, l_c=l_c, h_c=h_c100, X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_a0_lhs = thermdelay(inc=inc, m=m, a=a0, l_d=l_d_lhs, alpha=alpha, fsz=fsz, beta=beta, zeta=zeta, xi=xi, rhoStar_rho=rhoStar_rho, l_c=l_c, h_c=h_c10,  X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_a9_lhs = thermdelay(inc=inc, m=m, a=a9, l_d=l_d_lhs, alpha=alpha, fsz=fsz, beta=beta, zeta=zeta, xi=xi, rhoStar_rho=rhoStar_rho, l_c=l_c, h_c=h_c10,  X=X, Y=Y, Z=Z, GR_flag=GR_flag)
inst_a1_lhs = thermdelay(inc=inc, m=m, a=a1, l_d=l_d_lhs, alpha=alpha, fsz=fsz, beta=beta, zeta=zeta, xi=xi, rhoStar_rho=rhoStar_rho, l_c=l_c, h_c=h_c10,  X=X, Y=Y, Z=Z, GR_flag=GR_flag)

# Thermalization time delays
t_th_a0     = inst_a0.thermalization_time_delay(r=r0)
t_th_a9     = inst_a9.thermalization_time_delay(r=r9)
t_th_a1     = inst_a1.thermalization_time_delay(r=r1)
t_th_a0_lhs = inst_a0_lhs.thermalization_time_delay(r=r0)
t_th_a9_lhs = inst_a9_lhs.thermalization_time_delay(r=r9)
t_th_a1_lhs = inst_a1_lhs.thermalization_time_delay(r=r1)

# Light-travel time delays
t_lt_h1   = inst_h1.light_travel_time_delay(r=r1, phi=phi)[:,0]
t_lt_h10  = inst_h10.light_travel_time_delay(r=r1, phi=phi)[:,0]
t_lt_h100 = inst_h100.light_travel_time_delay(r=r1, phi=phi)[:,0]

# Thermal disk flux
TStar_eff_a0 = inst_a0.effective_temperature_at_effective_photosphere(r=r0)
TStar_eff_a9 = inst_a9.effective_temperature_at_effective_photosphere(r=r9)
TStar_eff_a1 = inst_a1.effective_temperature_at_effective_photosphere(r=r1)
F_th_a0      = c.sigma_sb.cgs * TStar_eff_a0.cgs**4
F_th_a9      = c.sigma_sb.cgs * TStar_eff_a9.cgs**4
F_th_a1      = c.sigma_sb.cgs * TStar_eff_a1.cgs**4

print("")
print(f"Peak thermalization time delay, t_th")
print(f"  a = {a0:.3f}: {np.max(t_th_a0).to('ms'):.2f}")
print(f"  a = {a9:.3f}: {np.max(t_th_a9).to('ms'):.2f}")
print(f"  a = {a1:.3f}: {np.max(t_th_a1).to('ms'):.2f}")
print("")
print(f"Radial width (R_g) at half-maximum t_th")
print(f"  a = {a0:.3f}: {(r0[t_th_a0 > np.max(t_th_a0)/2][-1] - r0[t_th_a0 > np.max(t_th_a0)/2][0]):.2f}")
print(f"  a = {a9:.3f}: {(r9[t_th_a9 > np.max(t_th_a9)/2][-1] - r9[t_th_a9 > np.max(t_th_a9)/2][0]):.2f}")
print(f"  a = {a1:.3f}: {(r1[t_th_a1 > np.max(t_th_a1)/2][-1] - r1[t_th_a1 > np.max(t_th_a1)/2][0]):.2f}")
print("")

#==============================================================================#
# Plotting options:
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathrsfs}')
opts_fig  = {'figsize':(8.4, 5.6), 'dpi':300}
opts_lrbt = {'left':0.15, 'right':0.85, 'bottom':0.20, 'top':0.97}

#==============================================================================#
# PLOT: Thermalization time delay (L axis) & Light-travel time delay (R axis)
fout = 'results/thermdelay.eps'

colorL, colorR = 'black', 'C3'
opts_xlab   = {'fontsize':28, 'ha':'center', 'va':'top'}
opts_ylabL  = {'fontsize':28, 'color':colorL, 'ha':'center', 'va':'center'}
opts_ylabR  = {'fontsize':28, 'color':colorR, 'ha':'center', 'va':'top'}
opts_xtmaj  = {'axis':'x', 'direction':'out', 'length':10, 'width':2,
              'labelsize':24, 'pad':5}
opts_xtmin  = {'axis':'x', 'direction':'out', 'length':5,  'width':2}
opts_ytmajL = {'axis':'y', 'direction':'out', 'length':10, 'width':2,
               'labelsize':24, 'pad':5, 'color':colorL, 'labelcolor':colorL}
opts_ytminL = {'axis':'y', 'direction':'out', 'length':5,  'width':2,
               'labelsize':24, 'pad':5, 'color':colorL, 'labelcolor':colorL}
opts_ytmajR = {'axis':'y', 'direction':'out', 'length':10, 'width':2,
               'labelsize':24, 'pad':5, 'color':colorR, 'labelcolor':colorR}
opts_ytminR = {'axis':'y', 'direction':'out', 'length':5,  'width':2,
               'labelsize':24, 'pad':5, 'color':colorR, 'labelcolor':colorR}
opts_plotL  = {'linewidth':2, 'color':colorL, 'zorder':2}
opts_plotR  = {'linewidth':2, 'color':colorR, 'zorder':2}

xlabel  = r'${\rm Disk\ Radius,}\ r \equiv R / R_{\rm g}$'
ylabelL = r'${\rm Thermalization\ Time,}\ t_{\rm th}\ [{\rm ms}]$'
ylabelR = r'${\rm Light}$'+'-'+r'${\rm Travel\ Time,}\ t_{\rm lt}\ [{\rm ms}]$'
xmin,  xmax   = 0, r_out
yminL, ymaxL  = 0.01, 200
yminR, ymaxR  = 0.01, 200

# Setup
fig = plt.figure(**opts_fig)
fig.subplots_adjust(**opts_lrbt)
axL = fig.add_subplot(111)
axR = axL.twinx()
axR.spines['right'].set_edgecolor(colorR)
axR.spines['left'].set_edgecolor(colorL)
# Labels
axL.set_xlabel(xlabel=xlabel,  **opts_xlab)
axL.set_ylabel(ylabel=ylabelL, **opts_ylabL)
axR.set_ylabel(ylabel=ylabelR, **opts_ylabR)
# Limits
axL.set_xlim(xmin,  xmax)
axR.set_xlim(xmin,  xmax)
axL.set_ylim(yminL, ymaxL)
axR.set_ylim(yminR, ymaxR)
# Scale
axL.set_xscale('linear')
axL.set_yscale('log')
axR.set_xscale('linear')
axR.set_yscale('log')
# Ticks
axL.tick_params(which='major', **opts_xtmaj)
axL.tick_params(which='minor', **opts_xtmin)
axL.tick_params(which='major', **opts_ytmajL)
axL.tick_params(which='minor', **opts_ytminL)
axR.tick_params(which='major', **opts_ytmajR)
axR.tick_params(which='minor', **opts_ytminR)
# Tick locations
yL_tick_locs = [ 0.01,   0.1,   1,   10,   100 ]
yL_tick_labs = ['0.01', '0.1', '1', '10', '100']
axL.set_yticks(yL_tick_locs, minor=False)
axL.set_yticklabels(yL_tick_labs, minor=False)
yR_tick_locs = [ 0.01,   0.1,   1,   10,   100 ]
yR_tick_labs = ['0.01', '0.1', '1', '10', '100']
axR.set_yticks(yR_tick_locs, minor=False)
axR.set_yticklabels(yR_tick_labs, minor=False)

# Plot: t_th [ms]
axL.plot(r0, t_th_a0.to(u.ms), **opts_plotL, linestyle='solid')
axL.plot(r9, t_th_a9.to(u.ms), **opts_plotL, linestyle='dashed')
axL.plot(r1, t_th_a1.to(u.ms), **opts_plotL, linestyle='dotted')

# Plot: t_lt [ms]
#axR.plot(r1, t_lt_h1.to(u.ms),   **opts_plotR, linestyle='solid')
axR.plot(r1, t_lt_h10.to(u.ms),  **opts_plotR, linestyle='solid')
#axR.plot(r1, t_lt_h100.to(u.ms), **opts_plotR, linestyle='dotted')

# Show where t_th > t_lt for a = 0.9
r9L = r9[t_th_a9.cgs > t_lt_h10.cgs][0]
r9R = r9[t_th_a9.cgs > t_lt_h10.cgs][-1]
axL.plot([r9L,r9L], [yminL,ymaxL], linewidth=1, color='Gray', zorder=1, linestyle='solid')
axL.plot([r9R,r9R], [yminL,ymaxL], linewidth=1, color='Gray', zorder=1, linestyle='solid')

print(f"t_th > t_lt for annulus with Rin = {r9L:.2f}, Rout = {r9R:.2f}")

#------------------------------------------------------------------------------#
# Inset
opts_xlabI  = {'fontsize':18, 'ha':'center', 'va':'bottom'}
opts_ylabI  = {'fontsize':18, 'color':colorL, 'ha':'center', 'va':'center'}
opts_xtmajI = {'axis':'x', 'direction':'out', 'length':5, 'width':2,
              'labelsize':14, 'pad':2.5}
opts_ytmajI = {'axis':'y', 'direction':'out', 'length':5,   'width':2,
               'labelsize':14, 'pad':2.5, 'color':colorL, 'labelcolor':colorL}
opts_plotI  = {'linewidth':2, 'color':colorL, 'zorder':2}

delta  = 0.015
aspect = (opts_lrbt['right'] - opts_lrbt['left']) \
       / (opts_lrbt['top'] - opts_lrbt['bottom'])
HI = 0.225
WI = HI * aspect
LI = opts_lrbt['right'] - WI - delta
BI = opts_lrbt['top'] - HI - delta * opts_fig['figsize'][0] / opts_fig['figsize'][1]
aspect_xy = (8.4 * 0.7) / (5.6 * 0.77)

axI = fig.add_axes([LI, BI, WI, HI])  # left, bottom, width, height
axI.set_xlabel(r"$r$", **opts_xlabI, labelpad=14)
axI.set_ylabel(r"$t_{\rm th}\ [{\rm ms}]$", **opts_ylabI, labelpad=8)
axI.set_xlim(0, r_out)
axI.set_ylim(1e-9, 2e-5)
axI.set_xscale('linear')
axI.set_yscale('log')
axI.tick_params(which='major', **opts_xtmajI)
axI.tick_params(which='major', **opts_ytmajI)
axI.minorticks_off()
axI.plot(r0, t_th_a0_lhs.to(u.ms), **opts_plotI, linestyle='solid')
axI.plot(r9, t_th_a9_lhs.to(u.ms), **opts_plotI, linestyle='dashed')
axI.plot(r1, t_th_a1_lhs.to(u.ms), **opts_plotI, linestyle='dotted')

#------------------------------------------------------------------------------#
'''
# Plot grid lines
for x in [0,10,20,30,40,50]:
    axL.plot([x,x], [yminL,ymaxL], linestyle='solid', linewidth=1, color='LightGray', zorder=0)
for y in yL_tick_locs:
    axL.plot([xmin,xmax], [y,y], linestyle='solid', linewidth=1, color='LightGray', zorder=0)
'''
# Fix the issue of different fonts
plt.rcParams.update({'font.family':'custom'})
# Save the figure
fig.savefig(fout, bbox_inches=0, dpi=opts_fig['dpi'])
plt.close()

#==============================================================================#
# PLOT:
fout = 'results/thermflux.eps'

color      = 'C0'
opts_xlab  = {'fontsize':28, 'ha':'center', 'va':'top'}
opts_ylab  = {'fontsize':28, 'color':color, 'ha':'center', 'va':'top'}
opts_xtmaj = {'axis':'x', 'direction':'out', 'length':10, 'width':2,
              'labelsize':24, 'pad':5}
opts_xtmin = {'axis':'x', 'direction':'out', 'length':5,  'width':2}
opts_ytmaj = {'axis':'y', 'direction':'out', 'length':10, 'width':2,
               'labelsize':24, 'pad':5, 'color':color, 'labelcolor':color}
opts_ytmin = {'axis':'y', 'direction':'out', 'length':5,  'width':2,
               'labelsize':24, 'pad':5, 'color':color, 'labelcolor':color}
opts_plot  = {'linewidth':2, 'color':color, 'zorder':2}

def flux2kTeff(F):
    F    *= u.erg / u.s / u.cm**2
    kTeff = c.k_B.cgs * (F.cgs / c.sigma_sb.cgs)**(1/4)
    return kTeff.to(u.keV)

def convert_axis(axL):
    y1, y2 = axL.get_ylim()
    axR.set_ylim(flux2kTeff(y1).to(u.keV).value, flux2kTeff(y2).to(u.keV).value)
    axR.figure.canvas.draw()

xlabel  = r'${\rm Disk\ Radius,}\ r \equiv R / R_{\rm g}$'
ylabelL = r'${\rm Thermal\ Flux,}\ F_{\rm r}^{\ast}\ [{\rm erg/s/cm^{2}}]$'
ylabelR = r'${\rm Eff.\ Temperature,}\ k T_{\rm eff}^{\ast}\ [{\rm keV}]$'
xmin,  xmax  = 0, r_out
yminL, ymaxL = 1e20, 2e24

# Setup
fig = plt.figure(**opts_fig)
fig.subplots_adjust(**opts_lrbt)
axL = fig.add_subplot(111)
axR = axL.twinx()
axR.spines['right'].set_edgecolor(color)
axR.spines['left'].set_edgecolor(color)

axL.callbacks.connect("ylim_changed", convert_axis)

# Labels
axL.set_xlabel(xlabel=xlabel,  **opts_xlab)
axL.set_ylabel(ylabel=ylabelL, **opts_ylab, labelpad=28)
axR.set_ylabel(ylabel=ylabelR, **opts_ylab, labelpad=20)
# Limits
axL.set_xlim(xmin,  xmax)
axR.set_xlim(xmin,  xmax)
axL.set_ylim(yminL, ymaxL)
# Scale
axL.set_xscale('linear')
axL.set_yscale('log')
axR.set_xscale('linear')
axR.set_yscale('log')
# Ticks
axL.tick_params(which='major', **opts_xtmaj)
axL.tick_params(which='minor', **opts_xtmin)
axL.tick_params(which='major', **opts_ytmaj)
axL.tick_params(which='minor', **opts_ytmin)
axR.tick_params(which='major', **opts_ytmaj)
axR.tick_params(which='minor', **opts_ytmin)
# Tick locations
yL_tick_locs = [1e20, 1e21, 1e22, 1e23, 1e24]
yL_tick_labs = [r'$10^{20}$', r'$10^{21}$', r'$10^{22}$', r'$10^{23}$', r'$10^{24}$']
axL.set_yticks(yL_tick_locs, minor=False)
axL.set_yticklabels(yL_tick_labs, minor=False)
yR_tick_locs_maj = [ 0.1,   1 ]
yR_tick_labs_maj = ['0.1', '1']
yR_tick_locs_min = [ 0.2,   0.3,   0.4,   0.5, 0.6, 0.7, 0.8, 0.9]
yR_tick_labs_min = ['0.2', '0.3', '0.4', '0.5', '',  '',  '',  '']
axR.set_yticks(yR_tick_locs_maj, minor=False)
axR.set_yticklabels(yR_tick_labs_maj, minor=False)
axR.set_yticks(yR_tick_locs_min, minor=True)
axR.set_yticklabels(yR_tick_labs_min, minor=True)

# Plot: F_th [erg/s/cm^2]
axL.plot(r0, F_th_a0.to(u.erg/u.s/u.cm**2), **opts_plot, linestyle='solid')
axL.plot(r9, F_th_a9.to(u.erg/u.s/u.cm**2), **opts_plot, linestyle='dashed')
axL.plot(r1, F_th_a1.to(u.erg/u.s/u.cm**2), **opts_plot, linestyle='dotted')
'''
# Plot: t_th (re-scaled)
ipeak_a0    = t_th_a0.argmax()
ipeak_a9    = t_th_a9.argmax()
ipeak_a1    = t_th_a1.argmax()
t_th_scl_a0 = (t_th_a0 / t_th_a0.max()) * F_th_a0[ipeak_a0]
t_th_scl_a9 = (t_th_a9 / t_th_a9.max()) * F_th_a9[ipeak_a9]
t_th_scl_a1 = (t_th_a1 / t_th_a1.max()) * F_th_a1[ipeak_a1]
opts_plot   = {'linewidth':2, 'color':'black', 'zorder':1}
#axL.plot(r0, t_th_scl_a0.to(u.erg/u.s/u.cm**2), **opts_plot, linestyle='solid')
axL.plot(r9, t_th_scl_a9.to(u.erg/u.s/u.cm**2), **opts_plot, linestyle='dashed')
#axL.plot(r1, t_th_scl_a1.to(u.erg/u.s/u.cm**2), **opts_plot, linestyle='dotted')

# Plot t_lt (re-scaled)
t_lt_scl_h10 = (t_lt_h10 / t_th_a9.max()) * F_th_a9[ipeak_a9]
opts_plot    = {'linewidth':2, 'color':'C3', 'zorder':1}
axL.plot(r1, t_lt_scl_h10.to(u.erg/u.s/u.cm**2), **opts_plot, linestyle='solid')
'''
# Show where t_th > t_lt for a = 0.9
opts_plot = {'linewidth':1, 'color':'Gray', 'zorder':1}
axL.plot([r9L,r9L], [yminL,ymaxL], **opts_plot, linestyle='solid')
axL.plot([r9R,r9R], [yminL,ymaxL], **opts_plot, linestyle='solid')
i9L = np.argmin(np.abs(r9 - r9L))
i9R = np.argmin(np.abs(r9 - r9R))
y9T = F_th_a9[i9L].to(u.erg/u.s/u.cm**2).value
y9B = F_th_a9[i9R].to(u.erg/u.s/u.cm**2).value
axL.plot([xmin,xmax], [y9T,y9T], **opts_plot, linestyle='solid')
axL.plot([xmin,xmax], [y9B,y9B], **opts_plot, linestyle='solid')
'''
# Plot grid lines
for x in [0,10,20,30,40,50]:
    axL.plot([x,x], [yminL,ymaxL], linestyle='solid', linewidth=1, color='LightGray', zorder=0)
for y in yL_tick_locs:
    axL.plot([xmin,xmax], [y,y], linestyle='solid', linewidth=1, color='LightGray', zorder=0)
'''
# Fix the issue of different fonts
plt.rcParams.update({'font.family':'custom'})
# Save the figure
fig.savefig(fout, bbox_inches=0, dpi=opts_fig['dpi'])
plt.close()


f_col = 1.7
print(f"{flux2kTeff(F=y9B):.2f}, {flux2kTeff(F=y9T):.2f}")
print(f"{flux2kTeff(F=y9B)*f_col:.2f}, {flux2kTeff(F=y9T)*f_col:.2f}")

#==============================================================================#
