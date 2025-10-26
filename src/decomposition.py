import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from typing import Tuple, List


def cp_decomposition(tensor: np.ndarray, rank: int, n_iter_max: int = 100) -> Tuple[np.ndarray, List[np.ndarray], float]:
    tl.set_backend('numpy')
    weights, factors = parafac(tensor, rank=rank, n_iter_max=n_iter_max, random_state=42)
    reconstructed = tl.cp_to_tensor((weights, factors))
    error = tl.norm(tensor - reconstructed, 2)
    return weights, factors, error


def tucker_decomposition(tensor: np.ndarray, ranks: Tuple[int, ...], 
                         n_iter_max: int = 100) -> Tuple[np.ndarray, List[np.ndarray], float]:
    tl.set_backend('numpy')
    core, factors = tucker(tensor, rank=ranks, n_iter_max=n_iter_max, random_state=42)
    reconstructed = tl.tucker_to_tensor((core, factors))
    error = tl.norm(tensor - reconstructed, 2)
    return core, factors, error


def reconstruct_from_cp(weights: np.ndarray, factors: List[np.ndarray]) -> np.ndarray:
    tl.set_backend('numpy')
    return tl.cp_to_tensor((weights, factors))


def reconstruct_from_tucker(core: np.ndarray, factors: List[np.ndarray]) -> np.ndarray:
    tl.set_backend('numpy')
    return tl.tucker_to_tensor((core, factors))
