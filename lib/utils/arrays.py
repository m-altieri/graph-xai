def set_value(a, axis, dim, value, update=False, copy=True, **kwargs):
    """Set value at a given axis and dimension(s) of a numpy array
    Args:
        a: numpy array
        axis: axis to modify, must be an integer
        dim: dimension(s) to modify, can be an integer or a list of integers
        value: value to set, its shape and type must be broadcastable to the
         shape and type of `a`
        update: if True, add value to the current value
        copy: if True, return a copy of the array
    Returns:
        a: numpy array with the value set
    """
    if copy:
        a = a.copy()
    idx = [slice(None) for _ in range(a.ndim)]
    idx[axis] = dim
    idx = tuple(idx)
    if update:
        a[idx] += value
    else:
        a[idx] = value
    return a


def get_value(a, axis, dim, **kwargs):
    """Get value at a given axis and dimension(s) of a numpy array
    Args:
        a: numpy array
        axis: axis to get value from, must be an integer
        dim: dimension(s) to get value from, can be an integer or a list of integers
    Returns:
        value: value at the given dimension and axis
    """
    idx = [slice(None) for _ in range(a.ndim)]
    idx[axis] = dim
    idx = tuple(idx)
    return a[idx]


def powerset(s, exclude_empty_list=True):
    """Return a list of all possible sublists within s.

    Args:
        s (list): list to compute the power set of.
        exclude_empty_list (bool): if True (default), the output will not contain the empty list `[]`.
    """
    x = len(s)
    res = []
    for i in range(1 << x):
        res.append([s[j] for j in range(x) if (i & (1 << j))])
    if exclude_empty_list:
        res.remove([])
    return res
