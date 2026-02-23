from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .utils import asarray_numeric

if TYPE_CHECKING:
    from .background import InflationHistory

__all__ = ["Scales"]


class Scales:

    def __init__(
        self,
        history: "InflationHistory",
        *,
        dtype=None,
    ):
        """
        Parameters
        ----------
        history : InflationHistory
            Background instance providing H(N) and pivot metadata.
        dtype : np.dtype or type, optional
            If None, uses history.dtype. Otherwise forces internal array dtype.
        """
        self.history = history
        self.dtype = history.dtype if dtype is None else dtype

        # Cache pivot reference values locally (as Python floats).
        self.N_pivot = float(history.N_pivot)
        self.k_pivot = float(history.k_pivot)
        self.H_pivot = float(history.H_pivot)


    # --------------------------
    # Core mappings
    # --------------------------
    def k_horizon_exit(self, N):
        """
        Comoving wavenumber that satisfies horizon exit at e-fold N:

            k_exit(N) = a(N) H(N)

        Using the pivot convention:
            a(N) = a_pivot * exp(N_pivot - N)
            k_pivot = a_pivot * H_pivot

        =>  k_exit(N) = k_pivot * exp(N_pivot - N) * (H(N)/H_pivot)

        Parameters
        ----------
        N : float or ndarray
            E-fold(s) before the end of inflation.

        Returns
        -------
        k_exit : ndarray
            Comoving horizon-exit scale(s) in Mpc^-1, same shape as N.
        """
        N_arr = asarray_numeric(N, dtype=self.dtype)
        HN = self.history.Hubble(N_arr)
        return self.k_pivot * np.exp(self.N_pivot - N_arr) * (HN / self.H_pivot)
    

    def k_at_horizon_ratio(self, N, ratio=0.005):
        """
        Comoving wavenumber such that the horizon ratio at e-fold N is:

            x(N,k) = k/(a(N)H(N)) = ratio

        Since k_exit(N) corresponds to ratio = 1:
            k(N, ratio) = ratio * k_exit(N)

        Parameters
        ----------
        N : float or ndarray
            E-fold(s) before the end of inflation.
        ratio : float or ndarray, default 0.005
            Desired value of k/(aH) at N. Small ratio corresponds to super-horizon
            evaluation (useful for power spectrum “freeze-out” evaluation).

        Returns
        -------
        k : ndarray
            Comoving wavenumber(s) in Mpc^-1. Broadcasts ratio against N.
        """
        N_arr = asarray_numeric(N, dtype=self.dtype)
        ratio_arr = asarray_numeric(ratio, dtype=self.dtype)
        return ratio_arr * self.k_horizon_exit(N_arr)


    def k_over_aH(self, N, k):
        """
        Dimensionless horizon ratio for fixed k:

            x(N,k) = k / (a(N) H(N))

        Parameters
        ----------
        N : float or ndarray
            E-fold(s) before the end of inflation.
        k : float or ndarray
            Comoving wavenumber(s) in Mpc^-1.

        Returns
        -------
        x : ndarray
            Dimensionless ratio k/(aH). Shape follows broadcasting rules.

        Broadcasting behavior
        ---------------------
        The implementation intentionally forms a broadcasted grid between N and k:
          - If N has shape (m,) and k has shape (K,), output is (m, K).
        """
        N_arr = asarray_numeric(N, dtype=self.dtype)
        k_arr = asarray_numeric(k, dtype=self.dtype)

        # Broadcast N against k:
        #   N2: (..., 1) and k2: (1, K) -> combined shape (..., K)
        N2 = N_arr[..., None]        # (..., 1)
        k2 = k_arr[None, ...]        # (1, ...)

        # H(N) matches N_arr shape; expand to (..., 1) for broadcasting with k-grid
        HN = self.history.Hubble(N_arr)  # shape (...) matching N_arr
        HN2 = HN[..., None]              # (..., 1)

        return (k2 / self.k_pivot) * np.exp(N2 - self.N_pivot) * (self.H_pivot / HN2)

