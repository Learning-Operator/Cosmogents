
The **Friedmann equations** are a set of fundamental equations in cosmology that describe the expansion of the universe based on general relativity and the cosmological principle (the assumption that the universe is homogeneous and isotropic at large scales). These equations govern the dynamics of the scale factor \(a(t)\), which measures the expansion of the universe.

The Friedmann equations come in different forms depending on the curvature of the universe. In a **flat universe**, the equation simplifies, and we can use it to describe the evolution of the scale factor \(a(t)\) over time.

### The First Friedmann Equation for a Flat Universe:

\[
\left( \frac{\dot{a}}{a} \right)^2 = \frac{8\pi G}{3} \rho - \frac{k c^2}{a^2} + \frac{\Lambda c}{3}
\]

Where:
- \( \dot{a} \) is the time derivative of the scale factor, representing the rate of change of the universe's size.
- \( a(t) \) is the scale factor as a function of time.

- \( \frac{\dot{a}}{a} \) is the hubble parameter, H.

- \( G \) is the gravitational constant.
- \( \rho \) is the energy density of the universe. (this can be broken down into matter density, radiation density and dark energy density, of which each scale differently as the universe increases in size. calculations for these are present in `cdensity.md`)
- \( k \) is the curvature parameter, which is zero for a flat universe.
- \(\Lambda\) is the cosmological constant (where the value of this constant can be found in `cosmo_constants.md` )
- \( c \) is the speed of light (where the value of this constant can be found in `cosmo_constants.md` )

For a flat universe (\(k = 0\)), this equation simplifies to:

\[
\left( \frac{\dot{a}}{a} \right)^2 = \frac{8\pi G}{3} \rho
\]

### The Second Friedmann Equation:

The second Friedmann equation involves the acceleration of the universe's expansion and is given by:

\[
\frac{\ddot{a}}{a} = - \frac{4\pi G}{3} (\rho + 3p)
\]

Where:
- \( \ddot{a} \) is the second derivative of the scale factor, representing the acceleration of the expansion.
- \( p \) is the pressure of the universe's contents.

### Cosmological Use of the Friedmann Equations

The Friedmann equations are used to model the expansion of the universe. They relate the scale factor \(a(t)\) to the energy content of the universe, including matter, radiation, and dark energy. By solving these equations with appropriate initial conditions, cosmologists can predict the evolution of the universe, including its expansion rate, the behavior of different components (such as dark energy and dark matter), and its ultimate fate (e.g., continuing expansion or collapse).

