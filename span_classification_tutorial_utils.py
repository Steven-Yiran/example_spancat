def to_dict(nl: list) -> dict:
    """
    Convert nested list to a dictionary for storing `.npz`

    Parameter
    ----------
    nl: list
        information in nested list structure

    Returns
    ---------
    d: dictionary
        dictionary with keys as index (converted to string) of the nested list, such that `d[str(i)] == nl[i]`.
    """
    return {str(i): l for i, l in enumerate(nl)}
