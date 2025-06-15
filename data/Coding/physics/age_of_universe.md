



The age of the universe, \(t_0\), can be estimated in two complementary ways: a zeroth-order “Hubble time” approximation and a more precise integral method within a flat ΛCDM model.

## Hubble Time Approximation

\[
t_H = \frac{1}{a * H_0}
\]

Where:  
- \(t_H\) is the Hubble time (approximate age of the universe)  
- \(H_0\) is the present-day Hubble constant (consult `hubble.md`)
- \(a\) is the scale factor (consult `cflrw.md`)

## Integral Method (ΛCDM Model)

\[
t_0 = \int_{0}^{\infty} \frac{dz}{(1 + z)\,H(z)}
\]

Where:  
- \(t_0\) is the age of the universe  
- \(z\) is redshift  
- \(H(z)\) is the Hubble parameter at redshift \(z\) (consult `hubble.md`)

### Hubble Parameter in a Flat ΛCDM Universe

\[
H(z) = H_0 \,\sqrt{\Omega_{m,0}(1+z)^3 \;+\;\Omega_{r,0}(1+z)^4 \;+\;\Omega_{\Lambda,0}}
\]

Where:  
- \(\Omega_{m,0}\) is the present-day matter density parameter (consult `density.md`)  
- \(\Omega_{r,0}\) is the present-day radiation density parameter (consult `density.md`)  
- \(\Omega_{\Lambda,0}\) is the present-day dark energy density parameter (consult `density.md`)

