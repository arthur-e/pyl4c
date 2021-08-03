L4C Calibration Extensions
==========================

The Dual Arrhenius-Michaelis Menten (DAMM) Model
------------------------------------------------

DAMM models (heterotrophic) respiration using two Michaelis-Menten coefficients, one for substrate availability and the other for oxygen availability:

$$
R_H = V_{max}\frac{[S_x]}{k_{M_{S_x}} + [S_x]}\frac{[O_2]}{k_{M_{O_2}} + [O_2]}
$$

Where $[S_x]$ and $[O_2]$ are the soluble substrate and oxygen concentrations, respectively, and $V_{max}$ is the maximum velocity of the reaction according to Arrhenius mechanics:

$$
V_{max} = \alpha\times \mathrm{exp}\left(\frac{-E_a}{RT}\right)
$$

Where $E_a$ is the activation energy of the enzymatic reaction, $R$ is the universal gas constant, and $T$ is the temperature in degrees K.

**There are two forms of the model, differing in the temperature-sensitivity of the Michaelis-Mentent coefficient for substrate availability.** In the constant-value form, $k_{M_{S_x}}$ is a scalar-valued free parameter. In the temperature-sensitive form, it is described by slope and intercept parameters:

$$
k_{M_{S_x}} = c_k + T_C m_k
$$

Where $T_C$ is the temperature in degrees Celsius.

**Concentrations:** Soluble substrate concentration is assumed to be fixed proportion $p$ of the total available soil carbon $C_{total}$ subject to diffusion:

$$
[S_x] = p[C_{total}]\times D_{liq}\times \theta^3
$$

Where "$\theta$ is the volumetric water content of the soil, and $D_{liq}$ is a diffusion coefficient of the substrate in liquid phase" (Davidson et al. 2012). In this form, $D_{liq}$ is a dimensionless diffusion coefficient that differs from what is traditionally obtained (in units of area or volume per unit time) using Fick's law.

The oxygen concentration is given by:

$$
[O_2] = D_{gas}\times 0.209\times \left[
1 - \frac{\rho_B}{\rho_P} - \theta
\right]^{4/3}
$$

Where 0.209 is the volume fraction of oxygen in the atmosphere. $\rho_B$ and $\rho_P$ are the bulk density and particle density of the soil, respectively, and subtracting the volumetric water content of the soil, $\theta$, provides the air-filled porosity. **Total porosity, $\phi$ is related to these soil densities by:**

$$
\phi = 1 - \frac{\rho_B}{\rho_P}
$$

Thus, we can make a substitution:

$$
[O_2] = D_{gas}\times 0.209\times \left[\phi - \theta\right]^{4/3}
$$

Model Parameters
----------------

**The free parameters of this system of equations to be fitted are:**

- $\alpha_i$, Pre-exponential factor of enzymatic reaction with $S_x$, for substrate pool $i$ (mg C cm-3 hr-1);
- $E_a$, Activation energy of enzymatic reaction with $S_x$ (kJ mol-1);
- $k_{M_{S_x}}$, Michaelis-Menten coefficient for substrate availability (g C cm-3);
- $p$, The proportion of C_total that is soluble;
- $D_liq$, Diffusion coefficient of substrate in liquid phase;
- $D_gas$, Diffusion coefficient of O_2 in air;

*There are as many $\alpha_i$ as there are substrate pools.*

**The Michaelis-Menten half-saturation parameter, $k_{M_{S_x}}$,** "increases more strongly with increasing temperature in cold-adapted enzymes than in warm-adapted enzymes...[it] is the substrate concentration at half-maximal enzymatic velocity (Vmax), and is indicative of the affinity an enzyme has for its substrate...
Therefore, an increase in [this parameter] indicates a decrease in overall enzyme function" German et al. (2012).

**The constants are:**

- $R$ = 8.314472e-3 (Universal gas constant, kJ K-1 mol-1)
- $k_{M_{O_2}}$ = 0.121 (sensitivity to available oxygen, constant w.r.t. temperature), "The value (0.121 cm3 O2 cm-3 air) is the value of [O2] used for kMO2 according to Eqn (6) at the mean value of soil moisture (0.229 cm3 H2Ocm-3 soil) in 2009 at this site." **NOTE: Could have this parameter fit to the data instead.**

Change in the substrate (SOC) pools is governed by the same equations as in L4C:
$$
\frac{dC_{met}}{dt} = \mathcal{L}f_{met} - R_{met}(t)
$$
$$
\frac{dC_{str}}{dt} = \mathcal{L}(1 - f_{met}) - R_{met}(t)
$$
$$
\frac{dC_{rec}}{dt} = f_{str}R_{str}(t) - R_{rec}(t)
$$


Calibration of DAMM Model
-------------------------

There are two key issues with calibrating the L4C-DAMM model:

1. Choosing which parameters should take on fixed values according to reasonable boundary conditions;
2. Determining the initial size of the substrate pools.

In addressing the first issue, we follow the examples of Davidson et al. (2012) and Sihi et al. (2018) and choose that:

- Under saturated conditions, $[S_x] = [S_{soluble}]$, which leads to the result: $D_{liq} = (1 / \theta^3)$, where $\theta$ is the VWC of the soil. We assume that the soil is saturate at or above the 95th percentile of VWC.
- Under completely dry conditions, the oxygen concentration of the soil is assumed to be the same as in free air, which leads to the result: $D_{gas} = 1 / a^{(4/3)}$. We assume that "completely" dry conditions occur at or below the 5th percentile of VWC.
- Finally, as model calibration seems to be improved by fixing the km_O2 parameter, as Davidson et al. (2012) and Sihi et al. (2018) do, we calculate km_O2 based on the median $D_gas$ and mean soil moisture (VWC) conditions across all sites.

**The average porosity at flux tower locations (0.45), according to the L4SM land model, is much lower than the average porosity in Davidson et al.'s (2012) dataset (0.68).**

In addressing the second issue, we following the example of L4C; assuming that the carbon substrate pools are in the steady state, we invert the respiration flux in order to estimate the steady-state soil carbon pool size(s). This amounts to solving the DAMM model for $C_{total}$.

$$
[C_{total}] = \bar{C} = \frac{k_{M_{S_x}}}{pD_{liq}\theta^3}
    \frac{R_H(k_{M_{O_2}} + [O_2])}{V_{max}[O_2] - R_H(k_{M_{O_2}} + [O_2])}
$$

It's advantageous to fit the model in this form (obtained through dividing the above by the denominator of the MM coefficient for O2), which is expressed in terms of the MM coefficient for O2:

$$
\bar{C} = \frac{R_H\, k_{M_{S_x}}}{(p\, D_{\mathrm{liq}}\, \theta^3)
    [V_{\mathrm{max}}\, [O_2](k_{M_{O_2}} + [O_2])^{-1} - R_H]}
$$
$$
\bar{C} = \frac{R_H\, k_{M_{S_x}}}{(p\, D_{\mathrm{liq}}\, \theta^3)
    (V_{\mathrm{max}}\, M_{O_2} - R_H)}
$$

**Calibration is conducted separately for each plant functional type (PFT).** This is based on the assumption that different plant-soil communities have fundamentally different responses to changes in temperature and moisture. For heterotrophic respiration specifically, it has been shown that soil microbial communities have locally adapted responses to changing temperatures (German et al. 2012).

**NOTES:**

- The bounds for carbon use efficiency (CUE) are based on the discussion of Cannell and Thornley (2000, p.50) and the review by Collalti and Prentice (2019).
- If V_max for the "structural" and "recalcitrant" pools (slowest 2) are too low, this can cause estimates of Cbar to be < 0.
- Using `cbar0()` (analytical solution) instead of `cbar()` (inversion) produces pretty similar fit statistics for PFT1, although the fitted parameter values are fairly different.


Initializing Substrate Pools
----------------------------

There are two approaches to initializing the substrate pools.

- With the `cbar()` method, **initial substrate pool sizes are estimated by inverting the observed tower respiration flux.** This does not require an initial estimate of daily litterfall. This is the "constant effect SOC factor" that L4C uses in calibration.
- With the `cbar0()` method, **initial substrate pool sizes are determined by an analytical solution of the differential equations governing change in substrate.** This is the steady-state value used in the "analytical spin-up" of SOC pools after calibration of L4C.

By setting the differential equations governing change in SOC to zero and solving for the (steady state) SOC stock.

$$
\frac{dC_{met}}{dt} &= \mathcal{L}f_{met} - R_{met}(t)\\
&= \mathcal{L}f_{met} - V_{max}^{[0]}\frac{pD_{liq}\theta^3[C_{total}]}{k_{M_{S_x}} + pD_{liq}\theta^3[C_{total}]}\frac{[O_2]}{k_{M_{O_2}} + [O_2]}\\
0 &= \mathcal{L}f_{met} - V_{max}^{[0]}\frac{\gamma[C_{total}]}{k_{M_{S_x}} + \gamma[C_{total}]}\beta\\
$$
$$
\frac{dC_{str}}{dt} &= \mathcal{L}(1 - f_{met}) - R_{str}(t)\\
&= \mathcal{L}(1 - f_{met}) - V_{max}^{[1]}\frac{pD_{liq}\theta^3[C_{total}]}{k_{M_{S_x}} + pD_{liq}\theta^3[C_{total}]}\frac{[O_2]}{k_{M_{O_2}} + [O_2]}\\
0 &= \mathcal{L}(1- f_{met}) - V_{max}^{[1]}\frac{\gamma[C_{total}]}{k_{M_{S_x}} + \gamma[C_{total}]}\beta
$$
$$
\frac{dC_{rec}}{dt} &= f_{str}R_{str}(t) - R_{rec}(t)\\
&= f_{str}V_{max}^{[1]}\frac{pD_{liq}\theta^3[C_{total}^{[1]}]}{k_{M_{S_x}} + pD_{liq}\theta^3[C_{total}^{[1]}]}\frac{[O_2]}{k_{M_{O_2}} + [O_2]} - V_{max}^{[2]}\frac{pD_{liq}\theta^3[C_{total}^{[2]}]}{k_{M_{S_x}} + pD_{liq}\theta^3[C_{total}^{[2]}]}\frac{[O_2]}{k_{M_{O_2}} + [O_2]}\\
0 &= f_{str}V_{max}^{[1]}\frac{\gamma[C_{total}^{[1]}]}{k_{M_{S_x}} + \gamma[C_{total}^{[1]}]}\beta + V_{max}^{[2]}\frac{\gamma[C_{total}^{[2]}]}{k_{M_{S_x}} + \gamma[C_{total}^{[2]}]}\beta\\
-f_{str}\frac{V_{max}^{[1]}}{V_{max}^{[2]}}\frac{\gamma[C_{total}^{[1]}]}{k_{M_{S_x}} + \gamma[C_{total}^{[1]}]} &= \frac{\gamma[C_{total}^{[2]}]}{k_{M_{S_x}} + \gamma[C_{total}^{[2]}]}\\
$$

**Solving for $C_{total}$ in each case, we obtain:**

$$
C_0 = \frac{f_\mathrm{met}\, \mathcal{L}\, k_{M_{S_x}}}{
    p\, D_{\mathrm{liq}}\, \theta^3\left(
       V_{\mathrm{max}}^{[0]}\, \beta -
       \mathcal{L}\, f_{\mathrm{met}} \right)}
\quad \mbox{where}\quad \beta = \frac{[O_2]}{k_{M_{O_2}} + [O_2]}
$$
C_1 = \frac{(1 - f_\mathrm{met})\, \mathcal{L}\, k_{M_{S_x}}}{
    p\, D_{\mathrm{liq}}\, \theta^3\,\left(
       V_{\mathrm{max}}^{[1]}\, \beta -
       \mathcal{L}\, (1 - f_\mathrm{met}) \right)}
$$
$$
C_2 = \frac{-V_{\mathrm{max}}^{[1]}\, C_1\, f_{\mathrm{str}}\, k_{M_{S_x}}}{
    p\,D_{liq}\,\theta^3\left(
      V_{\mathrm{max}}^{[1]}\, C_1\, f_{\mathrm{str}} +
      V_{\mathrm{max}}^{[2]}\, C_1\right) +
      V_{\mathrm{max}}^{[2]}\, k_{M_{S_x}}}
$$

Where $\mathcal{L}$ is average daily litterfall. **These steady-state C amounts are calculated for a climatological year, then the average in each pool is taken.**

**NOTE: Unfortunately, because the various $V_{max}$ are so small, C0, C1, and C2 end up converging to the same mean value.**

**Using these initial estimates, the mean steady-state SOC pool sizes are obtaining by finding the minimum of the absolute value of the annual NEE sum over a climatological year.** By minimizing this quantity, NPP inputs are balanced by RH outputs to within 5.8e-7 g C m-2 day-1. Because of the time complexity of minimization, we do this for the 26 Core Validation Sites at 9-km scale, not for the 26 x 81 model pixels. Thus, these steady-state values represent the spatial average of the 81 1-km pixels.


References
----------

Cannell, M. G. R., & Thornley, J. H. M. (2000). Modelling the components of plant respiration: Some guiding principles. Annals of Botany, 85, 45–54.

Collalti, A., & Prentice, I. C. (2019). Is NPP proportional to GPP? Waring’s hypothesis 20 years on. Tree Physiology, 39(8), 1473–1483.

German, D. P., K. R. B. Marcelo, M. M. Stone, and S. D. Allison. 2012. The Michaelis-Menten kinetics of soil extracellular enzymes in response to temperature: A cross-latitudinal study. Global Change Biology 18 (4):1468–1479.
