"""Exact arithmetic on native 4x bilinear rows. No numerical optimizer here."""
from fractions import Fraction as Q


def exact_row(proto, y, x):
    h, w = proto.shape[-2:]
    # align_corners=False at exactly 4x: (output+.5)/4-.5.
    def axis(v, length):
        n = 2 * int(v) - 3
        low = n // 8
        f = Q(n % 8, 8)
        return [(min(length-1, max(0, low)), 1-f),
                (min(length-1, max(0, low+1)), f)]
    return [sum((Q(float(proto[j, iy, ix])) * wy * wx
                 for iy, wy in axis(y, h) for ix, wx in axis(x, w)), Q(0))
            for j in range(proto.shape[0])]


def rationalize_farkas(rows, labels):
    import sympy as sp
    n = len(rows)
    if not n or n > 64:
        return None
    a = [[(1 if labels[i] else -1)*v for v in row] for i, row in enumerate(rows)]
    m = sp.Matrix([[sp.Rational(a[i][j].numerator, a[i][j].denominator)
                    for i in range(n)] for j in range(len(rows[0]))] +
                  [[int(y) for y in labels]])
    b = sp.Matrix([0]*len(rows[0]) + [1])
    try:
        sol, parameters = m.gauss_jordan_solve(b)
        if parameters.rows:
            sol = sol.subs({p: 0 for p in parameters})
        weights = [Q(int(v.p), int(v.q)) for v in sol]
    except (ValueError, ZeroDivisionError):
        return None
    if any(w < 0 for w in weights):
        return None
    if any(sum((weights[i]*a[i][j] for i in range(n)), Q(0)) != 0
           for j in range(len(rows[0]))):
        return None
    if sum((w*int(y) for w, y in zip(weights, labels)), Q(0)) != 1:
        return None
    return [str(w) for w in weights]


def margin_upper(rows, labels, weights):
    weights = [Q(w) for w in weights]
    if not rows or any(w < 0 for w in weights):
        raise ValueError('invalid margin dual weights')
    denom = sum((w*sum(map(abs, row)) for w, row in zip(weights, rows)), Q(0))
    if denom <= 0:
        raise ValueError('zero dual normalization')
    residual = [sum((w*(1 if y else -1)*row[j]
                     for w, y, row in zip(weights, labels, rows)), Q(0))
                for j in range(len(rows[0]))]
    return max(map(abs, residual))/denom


def verify(proto, certificate):
    rows = [exact_row(proto, *p) for p in certificate['pixels_yx']]
    labels = certificate['labels']
    weights = [Q(w) for w in certificate['weights']]
    if len(weights) != len(rows) or any(w < 0 for w in weights):
        raise AssertionError('invalid witness weights')
    if certificate['kind'] == 'farkas':
        assert sum((w*int(y) for w, y in zip(weights, labels)), Q(0)) == 1
        for j in range(len(rows[0])):
            assert sum((w*(1 if y else -1)*row[j]
                        for w, y, row in zip(weights, labels, rows)), Q(0)) == 0
    elif certificate['kind'] == 'margin_upper':
        assert margin_upper(rows, labels, certificate['weights']) == Q(certificate['upper_exact'])
    else:
        raise AssertionError('unknown certificate')
    return True
