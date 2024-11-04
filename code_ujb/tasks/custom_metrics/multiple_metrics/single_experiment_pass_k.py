import json

import numpy as np


def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def for_file(data, k=[1, 10, 100]):
    for r in data["results"]:
        if not "status" in r:
            print(r)
    n = len(data["results"])
    c = len(
        [True for r in data["results"] if r["status"] == "OK" and r["exit_code"] == 0]
    )
    return np.array([estimator(n, c, i) for i in k])
