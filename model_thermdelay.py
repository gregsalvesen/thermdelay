import numpy as np
import astropy.constants as c
import astropy.units as u
#==============================================================================#
'''
Functions to calculate the thermalization time delay from Salvesen (2022)
'''
#==============================================================================#
class thermdelay:

    #--------------------------------------------------------------------------#
    def __init__(self, inc, m, a, l_d, alpha, fsz, beta, zeta, xi, rhoStar_rho,
                 l_c, h_c, X, Y, Z, GR_flag):
        #......................................................................#
        # Input Quantities
        #......................................................................#
        # Observer parameters
        self.inc = inc.to(u.rad)  # Inclination angle (observer-to-BH)

        # Black hole parameters
        self.m = m  # BH mass, m = M / M_sun
        self.a = a  # BH spin

        # Disk parameters
        self.l     = l_d    # Disk luminosity (Eddington-scaled), l=L_disk/L_Edd
        self.alpha = alpha  # Effective viscosity parameter
        self.fsz   = fsz    # Frac of accretion power dissip in disk surf layers
        self.beta  = beta   # Albedo (reflectance) of the disk surface
        self.zeta  = zeta   # Param specifying form of hydrostatic equilib eqn
        self.xi    = xi     # Param specifying form of radiative diffusion eqn
        self.rhoStar_rho = rhoStar_rho  # Density ratio, eff photosp / mid-plane

        # Corona parameters
        self.l_c = l_c  # Corona luminosity (Eddington-scaled), l_c = L_c/L_Edd
        self.h_c = h_c  # Corona height, h_c = H_c / R_g

        # Other parameters
        self.X = X  # Hydrogen mass fraction
        self.Y = Y  # Helium mass fraction
        self.Z = Z  # Metals mass fraction
        if (X + Y + Z != 1):
            raise ValueError(f"`X` + `Y` + `Z` = {X+Y+Z}, but must be 1")

        # Turn relativistic corrections on/off (True/False)
        self.GR_flag = GR_flag

        #......................................................................#
        # Derived Quantities
        #......................................................................#
        # Electron scattering opacity
        kappa_es      = c.sigma_T.cgs / c.m_p.cgs * (X + Y/2 + Z/2)
        self.kappa_es = kappa_es.to(u.cm**2/u.g)
        
        # Eddington luminosity
        M          = m * c.M_sun.cgs
        L_Edd      = 4 * np.pi * c.G.cgs * M.cgs * c.c.cgs / kappa_es.cgs
        self.L_Edd = L_Edd.to(u.erg/u.s)
        
        # Gravitational radius
        R_g      = c.G.cgs * M.cgs / c.c.cgs**2
        self.R_g = R_g.to(u.cm)

        # Innermost stable circular orbit
        Z1          = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
        Z2          = np.sqrt(3 * a**2 + Z1**2)
        r_isco      = 3 + Z2 - np.sign(a) * np.sqrt((3 - Z1) * (3 + Z1 + 2*Z2))
        self.r_isco = r_isco

        # Radiative efficiency factor
        eta      = 1 - np.sqrt( 1 - 2 / (3 * r_isco) )
        self.eta = eta

        # Mass accretion rate (Eddington-scaled)
        mdot      = l_d / eta
        self.mdot = mdot

    #--------------------------------------------------------------------------#
    # Opacities
    
    def kramers_free_free_opacity(self, rho, T):
        kappa_ff = (4e22 * u.cm**2 / u.g) * (1 + self.X) * (1 - self.Z) \
                 * rho.cgs.value * T.cgs.value**(-7/2)
        return kappa_ff.to(u.cm**2/u.g)

    def kramers_bound_free_opacity(self, rho, T):
        kappa_bf = (4e25 * u.cm**2 / u.g) * (1 + self.X) * self.Z \
                 * rho.cgs.value * T.cgs.value**(-7/2)
        return kappa_bf.to(u.cm**2/u.g)

    def kramers_absorption_opacity(self, rho, T):
        kappa_th = self.kramers_free_free_opacity(rho=rho, T=T) \
                 + self.kramers_bound_free_opacity(rho=rho, T=T)
        return kappa_th.to(u.cm**2/u.g)
    
    #--------------------------------------------------------------------------#
    # GR correction factors
    def ABCD(self, r):
        if self.GR_flag:
            x  = np.sqrt(r)
            x0 = np.sqrt(self.r_isco)
            x1 =  2 * np.cos((np.arccos(self.a) - np.pi) / 3)
            x2 =  2 * np.cos((np.arccos(self.a) + np.pi) / 3)
            x3 = -2 * np.cos(np.arccos(self.a) / 3)
            A  = 1 - 2 / x**2 + self.a**2 / x**4
            B  = 1 - 3 / x**2 + 2 * self.a / x**3
            C  = 1 - 4 * self.a / x**3 + 3 * self.a**2 / x**4
            D  = 1 / x \
               * (x - x0 - 3 / 2 * self.a * np.log(x / x0)
                 - 3*(x1-self.a)**2/(x1*(x1-x2)*(x1-x3))*np.log((x-x1)/(x0-x1))
                 - 3*(x2-self.a)**2/(x2*(x2-x1)*(x2-x3))*np.log((x-x2)/(x0-x2))
                 - 3*(x3-self.a)**2/(x3*(x3-x1)*(x3-x2))*np.log((x-x3)/(x0-x3)))
        else:
            A, B, C = 1, 1, 1
            D       = 1 - np.sqrt(self.r_isco / r)
        return A, B, C, D
    
    #--------------------------------------------------------------------------#
    # Disk mid-plane radial structure

    def disk_midplane_scale_height(self, r):
        A, B, C, D = self.ABCD(r=r)
        H = 3 / 4 * c.G.cgs * c.M_sun.cgs / c.c.cgs**2 \
          * self.m * self.mdot * C**(-1) * D \
          * self.zeta * self.xi * (1 - self.fsz)
        return H.to(u.cm)  # H_d(r)

    def disk_midplane_density(self, r):
        A, B, C, D = self.ABCD(r=r)
        rho = 64/27 / self.kappa_es.cgs * c.c.cgs**2 / (c.G.cgs * c.M_sun.cgs) \
            * self.alpha**(-1) * self.m**(-1) * r**(3/2) * self.mdot**(-2) \
            * A**(-2) * B**(3/2) * C**(5/2) * D**(-2) \
            * self.zeta**(-2) * (self.xi * (1 - self.fsz))**(-3)
        return rho.to(u.g/u.cm**3)  # rho_d(r)
    
    def disk_midplane_optical_depth(self, r):
        A, B, C, D = self.ABCD(r=r)
        tau_es = 16 / 9 * self.alpha**(-1) * r**(3/2) * self.mdot**(-1) \
               * A**(-2) * B**(3/2) * C**(3/2) * D**(-1) \
               * self.zeta**(-1) * (self.xi * (1 - self.fsz))**(-2)
        return tau_es  # tau_es_d(r)

    def disk_midplane_pressure(self, r):
        A, B, C, D = self.ABCD(r=r)
        P = 4 / 3 * c.c.cgs**2 / self.kappa_es.cgs \
          * c.c.cgs**2 / (c.G.cgs * c.M_sun.cgs) \
          * self.alpha**(-1) * self.m**(-1) * r**(-3/2) \
          * A**(-2) * B**(1/2) * C**(3/2) \
          * (self.zeta * self.xi * (1 - self.fsz))**(-1)
        return P.to(u.dyn/u.cm**2)  # P_d(r)

    def disk_midplane_temperature(self, r):
        A, B, C, D = self.ABCD(r=r)
        T = (c.c.cgs**3 / c.sigma_sb.cgs / self.kappa_es.cgs)**(1/4) \
          * (c.c.cgs**2 / (c.G.cgs * c.M_sun.cgs))**(1/4) \
          * self.alpha**(-1/4) * self.m**(-1/4) * r**(-3/8) \
          * A**(-1/2) * B**(1/8) * C**(3/8) \
          * (self.zeta * self.xi * (1 - self.fsz))**(-1/4)
        return T.to(u.K)  #  T_d(r)

    #--------------------------------------------------------------------------#
    # Differential area of a disk ring
    def differential_disk_surface_area_dr(self, r):
        if self.GR_flag:
            if (np.sign(self.a) >= 0):
                dAring_dr = 2 * np.pi * self.R_g.cgs**2 \
                          * r**(1/4) * (r**(3/2) + self.a) \
                          / np.sqrt(r**(3/2) - 3 * np.sqrt(r) + 2 * self.a)
            if (np.sign(self.a) < 0):
                dAring_dr = 2 * np.pi * self.R_g.cgs**2 \
                          * r**(1/4) * (r**(3/2) - self.a) \
                          / np.sqrt(r**(3/2) - 3 * np.sqrt(r) - 2 * self.a)
        else:
            dAring_dr = 2 * np.pi * self.R_g.cgs**2 * r
        return dAring_dr.to(u.cm**2)  # [dAring/dr](r)

    #--------------------------------------------------------------------------#
    # Fluxes
    
    def disk_accretion_flux(self, r):
        A, B, C, D = self.ABCD(r=r)
        F_acc = 3 / 2 * c.c.cgs**3 / self.kappa_es.cgs \
              * c.c.cgs**2 / (c.G.cgs * c.M_sun.cgs) \
              * self.m**(-1) * r**(-3) * self.mdot * D / B
        return F_acc.to(u.erg/u.s/u.cm**2)  # F_acc(r)

    def irradiative_flux(self, r):
        dAring_dr = self.differential_disk_surface_area_dr(r=r)
        F_irr     = self.l_c * self.L_Edd.cgs \
                  / (4 * np.pi * self.R_g.cgs**2) \
                  * self.h_c / (self.h_c**2 + r**2)**(3/2) \
                  * 2 * np.pi * r * self.R_g.cgs**2 / dAring_dr.cgs
        return F_irr.to(u.erg/u.s/u.cm**2)  # F_irr(r)

    def irradiative_to_accretion_flux_ratio(self, r):
        F_acc = self.disk_accretion_flux(r=r)
        F_irr = self.irradiative_flux(r=r)
        F_rat = F_irr.cgs / F_acc.cgs
        return F_rat  # F_irr(r) / F_acc(r)

    #--------------------------------------------------------------------------#
    # Thermalization time delay
    def thermalization_time_delay(self, r):
        tau_es   = self.disk_midplane_optical_depth(r=r)
        rho      = self.disk_midplane_density(r=r)
        T        = self.disk_midplane_temperature(r=r)
        kappa_th = self.kramers_absorption_opacity(rho=rho.cgs, T=T.cgs)
        F_acc    = self.disk_accretion_flux(r=r)
        F_irr    = self.irradiative_flux(r=r)
        t_th     = 2 / c.c.cgs \
                 * self.kappa_es.cgs**(7/9) * kappa_th.cgs**(-16/9) \
                 * tau_es**(-14/9) * rho.cgs**(-1) * self.rhoStar_rho**(-25/9) \
                 * ((1+(1-self.beta)*F_irr.cgs/F_acc.cgs)/(1-self.fsz))**(14/9)
        return t_th.to(u.s)  # t_th(r)
    
    #--------------------------------------------------------------------------#
    # Color correction factor
    def color_correction(self, r):
        tau_es   = self.disk_midplane_optical_depth(r=r)
        rho      = self.disk_midplane_density(r=r)
        T        = self.disk_midplane_temperature(r=r)
        kappa_th = self.kramers_absorption_opacity(rho=rho.cgs, T=T.cgs)
        F_acc    = self.disk_accretion_flux(r=r)
        F_irr    = self.irradiative_flux(r=r)
        f_col    = (3/4)**(1/4) \
                 * self.kappa_es.cgs**(2/9) * kappa_th.cgs**(-2/9) \
                 * tau_es**(-7/36) * self.rhoStar_rho**(-2/9) \
                 * ((1+(1-self.beta)*F_irr.cgs/F_acc.cgs)/(1-self.fsz))**(7/36)
        return f_col  # f_col(r)
    
    #--------------------------------------------------------------------------#
    # T^*_eff
    def effective_temperature_at_effective_photosphere(self, r):
        F_acc     = self.disk_accretion_flux(r=r)
        F_irr     = self.irradiative_flux(r=r)
        TStar_eff = ((F_acc.cgs+(1-self.beta)*F_irr.cgs)/c.sigma_sb.cgs)**(1/4)
        return TStar_eff.to(u.K)  # T^*_eff(r)

    #--------------------------------------------------------------------------#
    # Light-travel time delay
    def light_travel_time_delay(self, r, phi):
        t_lt = self.R_g.cgs / c.c.cgs \
             * ( np.sqrt(r[:,None]**2 + self.h_c**2)
               - r[:,None] * np.sin(self.inc.cgs) * np.cos(phi[None,:].cgs)
               + self.h_c * np.cos(self.inc.cgs) )
        return t_lt.to(u.s)  # t_lt(r,phi)

#==============================================================================#
