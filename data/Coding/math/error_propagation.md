# Error Propagation Rules

In physical measurements and calculations, error propagation describes how uncertainties in individual variables affect the uncertainty in the result. Below are the key rules for propagating errors in different mathematical operations.

## 1. Addition or Subtraction

For two quantities \( a \) and \( b \), the propagated error when adding or subtracting them is given by:

\[
\Delta(a \pm b) = \Delta a + \Delta b
\]

Where:
- \( \Delta a \) and \( \Delta b \) are the uncertainties (errors) in \( a \) and \( b \), respectively.

**Note:** The uncertainty adds directly when adding or subtracting two values.



## 2. Multiplication

For the product of two quantities \( a \) and \( b \), the propagated relative error is given by:

\[
\frac{\Delta (a \cdot b)}{a \cdot b} = \frac{\Delta a}{a} + \frac{\Delta b}{b}
\]

Or equivalently:

\[
\Delta (a \cdot b) = \left( \frac{\Delta a}{a} + \frac{\Delta b}{b} \right) \cdot a \cdot b
\]

Where:
- \( \Delta a \) and \( \Delta b \) are the uncertainties in \( a \) and \( b \), respectively.



## 3. Division

For the quotient of two quantities \( a \) and \( b \), the propagated relative error is given by:

\[
\frac{\Delta (a / b)}{a / b} = \frac{\Delta a}{a} + \frac{\Delta b}{b}
\]

Or equivalently:

\[
\Delta (a / b) = \left( \frac{\Delta a}{a} + \frac{\Delta b}{b} \right) \cdot \frac{a}{b}
\]

Where:
- \( \Delta a \) and \( \Delta b \) are the uncertainties in \( a \) and \( b \), respectively.



## 4. Power of a Quantity

For a quantity \( a \) raised to the power of \( n \), the propagated error is given by:

\[
\frac{\Delta a^n}{a^n} = |n| \cdot \frac{\Delta a}{a}
\]

Or equivalently:

\[
\Delta a^n = |n| \cdot \frac{\Delta a}{a} \cdot a^n
\]

Where:
- \( \Delta a \) is the uncertainty in \( a \).
- \( n \) is the exponent to which \( a \) is raised.


