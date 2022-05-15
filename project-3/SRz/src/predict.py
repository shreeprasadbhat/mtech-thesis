def predict(equation) :
    expr = sympy.parse_expr(equation)

    symbols = sympy.symbols('u g r i z')

    fval = sympy.lambdify(symbols, expr, 'numpy')

    def fn(variables):
        return fval(*variables)

    return fn

if __name__ == '__main__':
    import numpy as np
    import sympy
    equation = ''
    y_pred = predict(equation)(variables)


