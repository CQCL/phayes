import sympy
from sympy import sin, cos, I

x = sympy.symbols("x")
a, b, c, d, e = sympy.symbols("a, b, c, d, e")

# fplus = a + b * cos(x) + c * sin(x) + d * cos(2*x) + e * sin(2*x)
# fminus = a - b * cos(x) - c * sin(x) + d * cos(2*x) + e * sin(2*x)

# f-(b) * (d/db f+(b))^2 - f+(b) * (d/db f-(b))^2
expr = (a - b * cos(x) - c * sin(x) + d * cos(2 * x) + e * sin(2 * x)) * (
    -b * sin(x) + c * cos(x) - 2 * d * sin(2 * x) + 2 * e * cos(2 * x)
) ** 2 - (a + b * cos(x) + c * sin(x) + d * cos(2 * x) + e * sin(2 * x)) * (
    b * sin(x) - c * cos(x) - 2 * d * sin(2 * x) + 2 * e * cos(2 * x)
) ** 2

expand_expr = sympy.expand(expr)
exptrig_expr = sympy.expand_trig(expand_expr)
expand_trig_expr = sympy.expand(exptrig_expr)
subs_sin_expr = expand_trig_expr.subs(sin(x) ** 2, 1 - cos(x) ** 2)
subs_sin_expr = subs_sin_expr.subs(sin(x) ** 3, (1 - cos(x) ** 2) * sin(x))
subs_sin_expr = sympy.expand(subs_sin_expr)
# sin_col_expr = sympy.collect(subs_sin_expr, sin(x))

lhs_coeffs = [subs_sin_expr.coeff(cos(x), i) for i in range(1, 4, 2)]
rhs_coeffs = [subs_sin_expr.coeff(cos(x), i).subs(sin(x), 1) for i in range(0, 4, 2)]


aprime, bprime = lhs_coeffs
cprime, dprime = rhs_coeffs

expr2 = (aprime * cos(x) + bprime * cos(x) ** 3) ** 2 - (1 - cos(x) ** 2) * (
    cprime + dprime * cos(x) ** 2
) ** 2
expand_expr2 = sympy.expand(expr2)

cos_coeffs = [expand_expr2.coeff(cos(x), i) for i in range(7)]

cos_coeffs_reversed = cos_coeffs[::-1]
