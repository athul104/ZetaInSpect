from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.special import hankel1, hankel2
from .utils import asarray_numeric
from .scales import Scales

if TYPE_CHECKING:
    from .background import InflationHistory

__all__ = ["Spectrum"]


class Spectrum:

    sqrt_pi_over_2 = np.sqrt(np.pi / 2.0)
    inv_4pi2 = 1.0 / (4.0 * np.pi * np.pi)

    def __init__(
        self,
        history: "InflationHistory",
        *,
        scales: Scales | None = None,
        dtype=None,
        enforce_hierarchy=True,
    ):
        self.history = history
        self.scales = Scales(history, dtype=dtype) if scales is None else scales
        self.dtype = history.dtype if dtype is None else dtype
        self.enforce_hierarchy = bool(enforce_hierarchy)
        if self.enforce_hierarchy and (not getattr(history, "hierarchy_ok", True)):
            raise ValueError(
                "Spectrum cannot be computed because the slow-roll hierarchy "
                "epsilon_H < |eta_H| is violated for this history. "
                "You can still use InflationHistory to inspect/plot H(N), epsilon_H(N), eta_H(N)."
            )

        # Cache arrays (small)
        self.eta = np.asarray(history.eta, dtype=self.dtype)  # (P,)
        self.transitions = np.asarray(history.transitions, dtype=self.dtype)  # (P-1,)
        self.P = int(self.eta.size)

        self._coeff_cache_fixed_k: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        # Precompute ν(η) per phase
        self.nu = self.nu_hankel(self.eta)  # (P,)

        # Precompute A_i per transition (i = 1..P-1) -- scalar per transition
        # Your Wands_duality case is preserved.
        if self.P > 1:
            eta_i = self.eta[1:]
            eta_im1 = self.eta[:-1]
            Wands_duality = (eta_i + eta_im1 == 3.0) & (eta_im1 > eta_i)
            A = eta_i - eta_im1
            A = np.where(Wands_duality, 0.0, A)
            self.A = A  # (P-1,)
        else:
            self.A = np.empty(0, dtype=self.dtype)

    # -------------------------
    # Small pure functions
    # -------------------------
    @staticmethod
    def nu_hankel(eta):
        """
        ν(η) for Hankel solutions.

        Parameters
        ----------
        eta : float or ndarray

        Returns
        -------
        nu : ndarray
            ν = sqrt(9/4 - 3η + η^2)
        """
        eta = np.asarray(eta)
        return np.sqrt(9.0 / 4.0 - 3.0 * eta + eta * eta)

    # -------------------------
    # Core: vk at fixed horizon ratio (k depends on N)
    # -------------------------
    def vk(self, N, ratio=0.005):
        """
        Evaluate Mukhanov–Sasaki variable v_k at a fixed horizon ratio x=ratio,
        where k is chosen per N such that x(N,k)=ratio.

        This reproduces your current workflow:
          k_N = k_at_horizon_ratio(N, ratio)
          propagate matching coefficients across phases up to phase_index(N)
          vk = sqrt(ratio) * [C1 H^(1)_ν(ratio) + C2 H^(2)_ν(ratio)]

        Parameters
        ----------
        N : float or ndarray
            E-fold(s) before the end.
        ratio : float, default 0.005
            Evaluation horizon ratio x=k/(aH). Small -> super-horizon evaluation.

        Returns
        -------
        vk : complex ndarray
            v_k evaluated at x=ratio, same shape as N.
        """
        N_arr = asarray_numeric(N, dtype=self.dtype)
        shape = N_arr.shape
        Nf = N_arr.ravel()

        # Phase index per N
        idxN = self.history.phase_index(Nf)  # (T,)
        # k(N) such that x(N,k)=ratio
        kN = np.asarray(self.scales.k_at_horizon_ratio(Nf, ratio), dtype=self.dtype)  # (T,)

        # Initial coefficients in phase 0
        nu0 = float(self.nu[0])
        C1 = (self.sqrt_pi_over_2 / np.sqrt(2.0 * kN)) * np.exp(1j * (nu0 + 0.5) * np.pi / 2.0)
        C2 = np.zeros_like(C1, dtype=np.complex128)

        # March through transitions i=1..P-1, update only points whose idxN >= i
        nu_prev = nu0
        for i in range(1, self.P):
            mask = (idxN >= i)
            if not np.any(mask):
                break

            nu_i = float(self.nu[i])
            A_i = float(self.A[i - 1]) if self.P > 1 else 0.0

            # Transition N value (scalar)
            N_tsn = float(self.transitions[i - 1])

            # T_tsn = x(N_tsn, kN)
            idx = mask.nonzero()[0]
            T_tsn = np.asarray(self.scales.k_over_aH(N_tsn, kN[idx]), dtype=self.dtype).reshape(-1)

            # Hankel evaluations at transition for current phase
            h1_i = hankel1(nu_i, T_tsn)
            h2_i = hankel2(nu_i, T_tsn)
            h1_i_m = hankel1(nu_i - 1.0, T_tsn)
            h2_i_m = hankel2(nu_i - 1.0, T_tsn)

            a1 = h1_i
            b1 = h2_i
            coeff_i = (0.5 - nu_i - A_i)
            a2 = coeff_i * h1_i + T_tsn * h1_i_m
            b2 = coeff_i * h2_i + T_tsn * h2_i_m

            # Hankel evaluations for previous phase
            h1_p = hankel1(nu_prev, T_tsn)
            h2_p = hankel2(nu_prev, T_tsn)
            h1_p_m = hankel1(nu_prev - 1.0, T_tsn)
            h2_p_m = hankel2(nu_prev - 1.0, T_tsn)

            d1 = h1_p * C1[idx] + h2_p * C2[idx]
            d2 = ((0.5 - nu_prev) * h1_p + T_tsn * h1_p_m) * C1[idx] + ((0.5 - nu_prev) * h2_p + T_tsn * h2_p_m) * C2[idx]

            denom = a1 * b2 - a2 * b1
            if np.any(np.isclose(denom, 0.0)):
                raise ZeroDivisionError("Denominator in (C1,C2) matching is ~0 for at least one N.")

            C1_new = (d1 * b2 - d2 * b1) / denom
            C2_new = (d2 * a1 - d1 * a2) / denom

            # Update only the points that have crossed this transition
            C1[idx] = C1_new
            C2[idx] = C2_new

            nu_prev = nu_i


        vkf = np.empty_like(C1, dtype=np.complex128)

        for i in range(self.P):
            sel = (idxN == i)
            if not np.any(sel):
                continue
            nu_i = float(self.nu[i])
            # hankel evaluated with scalar order and scalar argument
            h1 = hankel1(nu_i, ratio)
            h2 = hankel2(nu_i, ratio)
            vkf[sel] = np.sqrt(ratio) * (C1[sel] * h1 + C2[sel] * h2)

        return vkf.reshape(shape)
    
    

    def power_spectrum(self, N, ratio=0.005):
        """
        Primordial curvature power spectrum P_ζ evaluated at fixed horizon ratio x=ratio.

        Parameters
        ----------
        N : float or ndarray
            E-fold(s) before the end.
        ratio : float, default 0.005
            Evaluation horizon ratio x=k/(aH).

        Returns
        -------
        Pzeta : float ndarray
            Power spectrum values, same shape as N.
        """
        N_arr = asarray_numeric(N, dtype=self.dtype)

        eps = self.history.epsilon_H(N_arr)
        H = self.history.Hubble(N_arr)
        k = self.scales.k_at_horizon_ratio(N_arr, ratio)

        vk = self.vk(N_arr, ratio=ratio)
        vk2 = (vk.real * vk.real + vk.imag * vk.imag)


        Pzeta = k * self.inv_4pi2 * (H * H) * vk2 * (ratio * ratio) / eps
        return Pzeta

    # -------------------------
    # Tracking: vk(N) for fixed k
    # -------------------------
    def _coefficients_for_fixed_k(self, k):
        """
        Compute (C1, C2) coefficients at the start of each phase for a fixed comoving k.

        Performance notes
        -----------------
        - This is O(P) with Hankel evaluations at each transition (P is small).
        - Results are cached keyed by float(k) to avoid recomputation across repeated calls.

        Returns
        -------
        C1_phase, C2_phase : complex ndarray, shape (P,)
            Coefficients to be used in phase i.
        """
        k_key = float(k)
        cached = self._coeff_cache_fixed_k.get(k_key)
        if cached is not None:
            return cached

        C1_phase = np.empty(self.P, dtype=np.complex128)
        C2_phase = np.empty(self.P, dtype=np.complex128)

        nu0 = float(self.nu[0])
        C1 = (self.sqrt_pi_over_2 / np.sqrt(2.0 * k_key)) * np.exp(1j * (nu0 + 0.5) * np.pi / 2.0)
        C2 = 0.0 + 0.0j
        C1_phase[0] = C1
        C2_phase[0] = C2

        nu_prev = nu0
        for i in range(1, self.P):
            nu_i = float(self.nu[i])
            A_i = float(self.A[i - 1]) if self.P > 1 else 0.0
            N_tsn = float(self.transitions[i - 1])

            # scalar T at the transition for this fixed k
            T_tsn = np.asarray(self.scales.k_over_aH(N_tsn, k_key), dtype=self.dtype).reshape(-1)
            # make it a scalar (k is scalar so this should be length-1)
            T_tsn = float(T_tsn[0])

            # Current phase Hankels
            h1_i = hankel1(nu_i, T_tsn)
            h2_i = hankel2(nu_i, T_tsn)
            h1_i_m = hankel1(nu_i - 1.0, T_tsn)
            h2_i_m = hankel2(nu_i - 1.0, T_tsn)

            a1 = h1_i
            b1 = h2_i
            coeff_i = (0.5 - nu_i - A_i)
            a2 = coeff_i * h1_i + T_tsn * h1_i_m
            b2 = coeff_i * h2_i + T_tsn * h2_i_m

            # Previous phase Hankels
            h1_p = hankel1(nu_prev, T_tsn)
            h2_p = hankel2(nu_prev, T_tsn)
            h1_p_m = hankel1(nu_prev - 1.0, T_tsn)
            h2_p_m = hankel2(nu_prev - 1.0, T_tsn)

            d1 = h1_p * C1 + h2_p * C2
            d2 = ((0.5 - nu_prev) * h1_p + T_tsn * h1_p_m) * C1 + ((0.5 - nu_prev) * h2_p + T_tsn * h2_p_m) * C2

            denom = a1 * b2 - a2 * b1
            if np.isclose(denom, 0.0):
                raise ZeroDivisionError("Denominator in (C1,C2) matching is ~0 for fixed k.")

            C1 = (d1 * b2 - d2 * b1) / denom
            C2 = (d2 * a1 - d1 * a2) / denom

            C1_phase[i] = C1
            C2_phase[i] = C2
            nu_prev = nu_i

        self._coeff_cache_fixed_k[k_key] = (C1_phase, C2_phase)
        return C1_phase, C2_phase

    def _vk_track_with_T(self, N, k):
        """
        Internal: compute vk(N) and T(N)=k/(aH) for a fixed k.

        This is vectorized over N and loops only over phases (small P),
        evaluating Hankels only on the subset of points in each phase.
        """
        N_arr = asarray_numeric(N, dtype=self.dtype)
        shape = N_arr.shape
        Nf = N_arr.ravel()

        k_key = float(k)

        idxN = self.history.phase_index(Nf)  # (T,)
        T = np.asarray(self.scales.k_over_aH(Nf, k_key), dtype=self.dtype).reshape(-1)  # (T,)

        C1_phase, C2_phase = self._coefficients_for_fixed_k(k_key)

        vkf = np.empty_like(T, dtype=np.complex128)

        # Loop over phases (small) and evaluate Hankels only where needed
        for i in range(self.P):
            sel = (idxN == i)
            if not np.any(sel):
                continue

            T_sub = T[sel]
            nu_i = float(self.nu[i])

            vkf[sel] = np.sqrt(T_sub) * (
                C1_phase[i] * hankel1(nu_i, T_sub) + C2_phase[i] * hankel2(nu_i, T_sub)
            )

        return vkf.reshape(shape), T.reshape(shape)

    def vk_track(self, N, k):
        """
        v_k(N) as a function of N for a fixed comoving wavenumber k.

        Vectorized over N. Uses cached phase coefficients for this k.
        """
        vk, _T = self._vk_track_with_T(N, k)
        
        return vk

    def power_spectrum_track(self, N, k):
        """
        P_ζ(N) for a fixed comoving mode k.

        Vectorized over N. Reuses the same T(N)=k/(aH) used in vk_track.
        """
        N_arr = asarray_numeric(N, dtype=self.dtype)

        eps = self.history.epsilon_H(N_arr)
        H = self.history.Hubble(N_arr)

        vk, T = self._vk_track_with_T(N_arr, k)
        vk2 = (vk.real * vk.real + vk.imag * vk.imag)

        Pzeta = float(k) * self.inv_4pi2 * (H * H) * vk2 * (T * T) / eps

        return Pzeta