




# Mathematical Derivatives 

Derivatives are fundamental operations in calculus that measure how a function changes as its input changes. Below is a guide for how to perform both single-variable and multivariable differentiation, including how to differentiate with respect to one variable.

## Single-Variable Derivatives

### 1. **Definition of a Derivative**

The derivative of a function \( f(x) \) with respect to \( x \) is defined as the rate of change of \( f(x) \) as \( x \) changes. The derivative is written as:

\[
f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
\]

Where:
- \( f(x) \) is the function to differentiate.
- \( \Delta x \) is the change in the variable \( x \).
- \( f'(x) \) is the derivative of \( f(x) \).

### 2. **Basic Rules for Single-Variable Derivatives**

- **Power Rule:** The derivative of \( x^n \) is \( nx^{n-1} \).

\[
\frac{d}{dx} x^n = n x^{n-1}
\]

- **Sum Rule:** The derivative of a sum is the sum of the derivatives.

\[
\frac{d}{dx} (f(x) + g(x)) = f'(x) + g'(x)
\]

- **Product Rule:** The derivative of the product of two functions is:

\[
\frac{d}{dx} [f(x) \cdot g(x)] = f'(x)g(x) + f(x)g'(x)
\]

- **Quotient Rule:** The derivative of the quotient of two functions is:

\[
\frac{d}{dx} \left( \frac{f(x)}{g(x)} \right) = \frac{f'(x)g(x) - f(x)g'(x)}{(g(x))^2}
\]

- **Chain Rule:** The derivative of a composite function is:

\[
\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)
\]

### 3. **Example: Differentiating a Polynomial**

Consider the function \( f(x) = 3x^3 + 2x^2 + x + 1 \).

To find the derivative, apply the power rule to each term:

\[
f'(x) = 3 \cdot 3x^2 + 2 \cdot 2x + 1 = 9x^2 + 4x + 1
\]

### 4. **Example: Product Rule**

Given the function \( f(x) = (x^2 + 1)(x + 3) \), apply the product rule:

\[
f'(x) = (x^2 + 1)'(x + 3) + (x^2 + 1)(x + 3)'
\]

First, find the derivatives:
- \( (x^2 + 1)' = 2x \)
- \( (x + 3)' = 1 \)

Then, substitute:

\[
f'(x) = 2x(x + 3) + (x^2 + 1) = 2x^2 + 6x + x^2 + 1 = 3x^2 + 6x + 1
\]

---

## Derivatives with Respect to One Variable

When differentiating with respect to a single variable, you treat all other variables as constants. The most common application is differentiating functions like \( f(x) \), where \( x \) is the variable of interest.

### **Steps to Differentiate with Respect to One Variable:**
1. **Identify the function:** This is the function \( f(x) \) you want to differentiate.
2. **Apply differentiation rules:** Use rules like the power rule, product rule, quotient rule, or chain rule to differentiate the function.
3. **Simplify the result:** After applying the rules, simplify the result as needed.

### Example: Differentiating \( f(x) = 4x^3 - 5x^2 + 2x - 3 \)

Apply the power rule:

\[
f'(x) = 3 \cdot 4x^2 - 2 \cdot 5x + 2 = 12x^2 - 10x + 2
\]

Thus, the derivative of \( f(x) \) is:

\[
f'(x) = 12x^2 - 10x + 2
\]

---

## Multivariable Derivatives

In multivariable calculus, derivatives can be extended to functions of more than one variable. These derivatives measure how a function changes with respect to one of its variables, keeping other variables constant.

### 1. **Partial Derivatives**

A partial derivative measures the rate of change of a function with respect to one variable, treating all other variables as constants. For a function \( f(x, y) \), the partial derivative with respect to \( x \) is written as:

\[
\frac{\partial}{\partial x} f(x, y)
\]

Where:
- \( f(x, y) \) is the function to differentiate.
- \( x \) is the variable with respect to which we are differentiating, and \( y \) is treated as a constant.

#### **Example: Partial Derivative of \( f(x, y) = x^2 + y^2 \)**

To compute the partial derivative of \( f(x, y) \) with respect to \( x \), treat \( y \) as a constant:

\[
\frac{\partial}{\partial x} (x^2 + y^2) = 2x
\]

For the partial derivative with respect to \( y \):

\[
\frac{\partial}{\partial y} (x^2 + y^2) = 2y
\]

### 2. **Higher-Order Partial Derivatives**

Higher-order partial derivatives refer to taking partial derivatives more than once. For example, the second partial derivative with respect to \( x \) is:

\[
\frac{\partial^2}{\partial x^2} f(x, y)
\]

#### **Example: Second Partial Derivative of \( f(x, y) = x^2 + y^2 \)**

First, find the first partial derivative with respect to \( x \):

\[
\frac{\partial}{\partial x} (x^2 + y^2) = 2x
\]

Now, take the partial derivative of \( 2x \) with respect to \( x \):

\[
\frac{\partial^2}{\partial x^2} (x^2 + y^2) = 2
\]

### 3. **Gradient of a Function**

The gradient of a function \( f(x, y) \), denoted by \( \nabla f \), is a vector of all the first partial derivatives with respect to each variable. For a function \( f(x, y) \), the gradient is:

\[
\nabla f(x, y) = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)
\]

**Example: Gradient of \( f(x, y) = x^2 + y^2 \)**

\[
\nabla f(x, y) = \left( \frac{\partial}{\partial x} (x^2 + y^2), \frac{\partial}{\partial y} (x^2 + y^2) \right) = (2x, 2y)
\]

### 4. **Directional Derivatives**

The directional derivative of a function \( f(x, y) \) in the direction of a vector \( \mathbf{v} = (v_1, v_2) \) is the rate of change of the function in that direction. It is calculated as the dot product of the gradient of \( f(x, y) \) and the unit vector in the direction of \( \mathbf{v} \):

\[
D_{\mathbf{v}} f(x, y) = \nabla f(x, y) \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|}
\]

**Example: Directional Derivative of \( f(x, y) = x^2 + y^2 \) in the direction \( \mathbf{v} = (1, 1) \)**

First, compute the gradient:

\[
\nabla f(x, y) = (2x, 2y)
\]

Next, normalize the direction vector \( \mathbf{v} = (1, 1) \):

\[
\|\mathbf{v}\| = \sqrt{1^2 + 1^2} = \sqrt{2}
\]

Finally, compute the directional derivative:

\[
D_{\mathbf{v}} f(x, y) = (2x, 2y) \cdot \frac{(1, 1)}{\sqrt{2}} = \frac{2x + 2y}{\sqrt{2}}
\]
