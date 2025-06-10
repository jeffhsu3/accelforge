import sympy


def Max(a, *bs):
    """More post-lambdify broadcast-friendly option than sympy.Max"""
    result = a
    for b in bs:
        if isinstance(result, sympy.Expr) and isinstance(b, sympy.Expr):
            result = sympy.Max(result, b)
        else:
            result = sympy.Piecewise((result, result > b), (b, True))
    return result
