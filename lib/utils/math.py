import math

def divide_into_3_integers(n, p1, p2, p3):
    """n: integer to divide.\npx: percentage of each resulting number relative to n.
    (p1 + p2 + p3) is assumed to be 1.
    Returns 3 integers, whose sum is n, and that are within 1 from n * px.
    """
    n1 = math.floor(n * p1)
    n2 = math.floor(n * p2)
    n3 = math.floor(n * p3)
    r = n - n1 - n2 - n3
    if r == 1: n1 += 1
    if r == 2:
        n2 += 1
        n3 += 1
    return n1, n2, n3
