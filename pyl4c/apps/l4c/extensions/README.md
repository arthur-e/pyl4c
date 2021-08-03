The steady-state equations for the three C pools of the combined L4C-DAMM
model are:

$$
C_0 &= \frac{f_\mathrm{met}\, \mathcal{L}\, k_{M_{S_x}}}{
    p\, D_{\mathrm{liq}}\, \theta^3\left(
       V_{\mathrm{max}}^{[0]}\, \beta -
       \mathcal{L}\, f_{\mathrm{met}} \right)}
\quad \mbox{where}\quad \beta = \frac{[O_2]}{k_{M_{O_2}} + [O_2]}\\
C_1 &= \frac{(1 - f_\mathrm{met})\, \mathcal{L}\, k_{M_{S_x}}}{
    p\, D_{\mathrm{liq}}\, \theta^3\,\left(
       V_{\mathrm{max}}^{[1]}\, \beta -
       \mathcal{L}\, (1 - f_\mathrm{met}) \right)}\\
C_2 &= \frac{-V_{\mathrm{max}}^{[1]}\, C_1\, f_{\mathrm{str}}\, k_{M_{S_x}}}{
    p\,D_{liq}\,\theta^3\left(
      V_{\mathrm{max}}^{[1]}\, C_1\, f_{\mathrm{str}} +
      V_{\mathrm{max}}^{[2]}\, C_1\right) +
      V_{\mathrm{max}}^{[2]}\, k_{M_{S_x}}}
$$
