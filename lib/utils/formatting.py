import numpy as np


def indent(depth=1):
    """Create an indentation string of the specified indentation depth.

    :param depth: Indentation depth, defaults to 1. Each unit of indentation is two characters wide.
    :type depth: int, optional
    :return: Indentation string.
    :rtype: str
    """
    return f"".join(["--" for i in range(depth - 1)]) + "> "


class NumpyFullprint:
    def __enter__(self):
        self.t = np.get_printoptions()["threshold"]
        np.set_printoptions(threshold=np.inf)

    def __exit__(self, *args, **kwargs):
        np.set_printoptions(threshold=self.t)
