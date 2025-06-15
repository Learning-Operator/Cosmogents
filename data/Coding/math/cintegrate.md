


# Mathematical Integration 

Integration is a fundamental operation in calculus used to find quantities like areas, volumes, or accumulated changes. Below is a guide for how to perform both single-variable and multivariable integration, including how to integrate with respect to a single variable.

## Single-Variable Integration

### 1. **Indefinite Integrals**

Indefinite integrals represent the antiderivative of a function, meaning they reverse the process of differentiation. The integral of a function \( f(x) \) is written as:

\[
\int f(x) \, dx = F(x) + C
\]

Where:
- \( f(x) \) is the function to integrate.
- \( F(x) \) is the antiderivative of \( f(x) \).
- \( C \) is the constant of integration.

**Steps:**
- Identify the function \( f(x) \) you need to integrate.
- Apply the integration rules (e.g., power rule, trigonometric identities, etc.).
- Add the constant of integration \( C \).

**Example:**
For \( f(x) = x^2 \), apply the power rule of integration:

\[
\int x^2 \, dx = \frac{x^3}{3} + C
\]

### 2. **Definite Integrals**

Definite integrals calculate the total accumulation of a quantity over a specific interval \([a, b]\). The integral is written as:

\[
\int_a^b f(x) \, dx = F(b) - F(a)
\]

Where:
- \( a \) and \( b \) are the limits of integration.
- \( F(x) \) is the antiderivative of \( f(x) \).

**Steps:**
- Find the indefinite integral of \( f(x) \) (i.e., find \( F(x) \)).
- Evaluate \( F(x) \) at the upper and lower limits, then subtract.

**Example:**
For \( f(x) = x^2 \), and limits \( a = 0 \) and \( b = 2 \), the definite integral is:

\[
\int_0^2 x^2 \, dx = \left[ \frac{x^3}{3} \right]_0^2 = \frac{8}{3} - 0 = \frac{8}{3}
\]



## Integration with Respect to One Variable

When integrating with respect to a single variable, you treat all other variables in the function as constants. This is the typical form for single-variable integration, where we integrate a function with respect to \( x \) (or another variable).

### **Steps to Integrate with Respect to One Variable:**
1. **Identify the variable of integration:** In most cases, this will be \( x \), but could be any variable depending on the context.
2. **Rewrite the function to simplify integration:** Often, you will need to manipulate the function (e.g., expand polynomials, simplify trigonometric expressions).
3. **Apply the appropriate integration rules:** Use standard integration techniques (e.g., power rule, trigonometric integrals, substitution, etc.).
4. **Add the constant of integration** when performing an indefinite integral.

### Example: Integration with respect to \( x \)

Given the function \( f(x) = 3x^2 + 2x \), you would integrate with respect to \( x \):

\[
\int (3x^2 + 2x) \, dx
\]

1. Apply the power rule for integration:
   - \( \int x^n \, dx = \frac{x^{n+1}}{n+1} + C \)
   - For \( 3x^2 \), the integral is \( \frac{3x^3}{3} = x^3 \).
   - For \( 2x \), the integral is \( \frac{2x^2}{2} = x^2 \).

2. The result is:
   \[
   \int (3x^2 + 2x) \, dx = x^3 + x^2 + C
   \]

### Example: Definite Integration with respect to \( x \)

Given the function \( f(x) = 4x + 1 \) with bounds \( a = 1 \) and \( b = 3 \), you would calculate the definite integral:

\[
\int_1^3 (4x + 1) \, dx
\]

1. First, find the indefinite integral:
   \[
   \int (4x + 1) \, dx = 2x^2 + x + C
   \]

2. Now, evaluate the result at the upper and lower bounds:
   \[
   \left[ 2x^2 + x \right]_1^3 = (2(3)^2 + 3) - (2(1)^2 + 1)
   \]
   \[
   = (18 + 3) - (2 + 1) = 21 - 3 = 18
   \]

So, the definite integral is 18.


## Multivariable Integration

Multivariable integration extends the concept of integration to functions of more than one variable. It is often used to calculate areas, volumes, or mass distributions in higher dimensions.

### 1. **Double Integrals**

Double integrals are used to integrate functions of two variables over a two-dimensional region.

The double integral is written as:

\[
\int \int_R f(x, y) \, dA
\]

Where:
- \( R \) is the region of integration in the \( xy \)-plane.
- \( f(x, y) \) is the function to integrate.
- \( dA \) is the area differential (typically \( dx \, dy \) or \( dy \, dx \)).

**Steps:**
1. Set up the bounds of integration for both \( x \) and \( y \).
2. Integrate with respect to one variable (usually \( x \) first).
3. Integrate with respect to the second variable (usually \( y \) second).

**Example:**
For \( f(x, y) = x + y \), over a square region with bounds \( 0 \leq x \leq 1 \) and \( 0 \leq y \leq 1 \), the double integral is:

\[
\int_0^1 \int_0^1 (x + y) \, dx \, dy
\]

1. First, integrate with respect to \( x \):
\[
\int_0^1 (x + y) \, dx = \left[ \frac{x^2}{2} + xy \right]_0^1 = \frac{1}{2} + y
\]

2. Now, integrate with respect to \( y \):
\[
\int_0^1 \left( \frac{1}{2} + y \right) \, dy = \left[ \frac{y}{2} + \frac{y^2}{2} \right]_0^1 = \frac{1}{2} + \frac{1}{2} = 1
\]

### 2. **Triple Integrals**

Triple integrals extend the double integral to three variables, often used to calculate the volume under a surface or mass within a 3D region.

The triple integral is written as:

\[
\int \int \int_V f(x, y, z) \, dV
\]

Where:
- \( V \) is the three-dimensional region of integration.
- \( f(x, y, z) \) is the function to integrate.
- \( dV \) is the volume differential.

**Steps:**
1. Set up the bounds of integration for all three variables.
2. Integrate with respect to one variable, then the second, and finally the third.

**Example:**
For a function \( f(x, y, z) = x + y + z \), the triple integral over a cube with bounds \( 0 \leq x \leq 1 \), \( 0 \leq y \leq 1 \), and \( 0 \leq z \leq 1 \) is:

\[
\int_0^1 \int_0^1 \int_0^1 (x + y + z) \, dx \, dy \, dz
\]

1. First, integrate with respect to \( x \):
\[
\int_0^1 (x + y + z) \, dx = \left[ \frac{x^2}{2} + xy + xz \right]_0^1 = \frac{1}{2} + y + z
\]

2. Next, integrate with respect to \( y \):
\[
\int_0^1 \left( \frac{1}{2} + y + z \right) \, dy = \left[ \frac{y}{2} + \frac{y^2}{2} + yz \right]_0^1 = \frac{1}{2} + \frac{1}{2} + z = 1 + z
\]

3. Finally, integrate with respect to \( z \):
\[
\int_0^1 (1 + z) \, dz = \left[ z + \frac{z^2}{2} \right]_0^1 = 1 + \frac{1}{2} = \frac{3}{2
