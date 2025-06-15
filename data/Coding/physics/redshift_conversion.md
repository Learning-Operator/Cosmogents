

Redshift is a measure of recessional velocity. Beyond that, it is also a measure of the distance to which an object is farther from an observer, given the expansion of our universe.

## Hubble's Law

\[
z = H_0 \cdot d
\]

Where:
- \( z \) is the redshift
- \( H_0 \) is the Hubble parameter
- \( d \) is the proper distance (consult `proper_dist.md` to find calculations for it)

### Redshift and the Size of the Universe

Redshift also has strong relations with the size of our universe.

\[
a = \frac{1}{1 + z}
\]

Where:
- \( a \) is the scale factor (a measure of the size of the universe, in comparison to that of today, where \( a = 1 \))

### To Obtain Redshift from the Scale Factor

\[
z = \frac{1}{a} - 1
\]

### To Calculate Cosmic Time

The cosmic time is given by the integral:

\[
t(z) = \frac{1}{H_0} \int_0^\infty \frac{1}{((1 + z) * \sqrt{\Omega (1 + z)^3 + \Omega_\Lambda})} \, dz
\]

This integral is called Hogg's Expression for "look back time"

*The integral calculation can be found in `cintegrate.md`.*

Where:
- \( t(z) \) is the cosmic time
- \( H(z) \) is the Hubble parameter (its calculations are presented in `chubble.md`)
