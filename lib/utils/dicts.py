def dict_union(*dicts, overwrite_none=False):
    """Perform a non-commutative (left-to-right incremental) dict union.
    The result will have, for each common key, `{'k': 'v1'}, ..., {'k: 'vn'} -> {'k': 'vn'}`.

    Args:
        overwrite_none (bool, optional): If True, `None` values with on the same key will be replaced:
            `dict_union({'k': 'v'}, {'k': None}) == {'k': None}`, if False, they will be skipped.
            Defaults to False.

    Returns:
        dict: The union of the input dicts.
    """
    output_dict = {}
    for d in dicts:
        output_dict = output_dict | {
            k: d[k] for k in d if d[k] is not None or overwrite_none
        }
    return output_dict
