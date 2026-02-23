from __future__ import annotations
import numpy as np
import warnings
from .utils import asarray_numeric, require_strictly_decreasing, isclose_zero

__all__ = ["InflationHistory"]


class InflationHistory:

    def __init__(
        self,
        eta_list,
        efold_list,
        r,
        A_s=2.1e-9,
        *,
        N_pivot=60.0,
        N_max=65.0,
        N_min=0.5,
        k_pivot=0.05,
        dtype=np.float64,
    ):
        """
        Parameters
        ----------
        eta_list : array-like of float, shape (P,)
            Values of eta_H for each phase (piecewise constant).
        efold_list : array-like of float, shape (P-1,)
            Transition e-fold values (descending) separating phases.
        r : float
            Tensor-to-scalar ratio at the pivot scale.
        A_s : float, default 2.1e-9
            Scalar power spectrum amplitude at the pivot scale.

        Keyword-only parameters
        -----------------------
        N_pivot : float, default 60.0
            Pivot scale e-fold value.
        N_max : float, default 65.0
            Maximum e-fold value allowed (start of history domain).
        N_min : float, default 0.5
            Minimum e-fold value allowed (end of history domain).
        k_pivot : float, default 0.05
            Pivot scale comoving wavenumber in Mpc^-1 (stored for scale conversions).
        dtype : numpy dtype, default np.float64
            Floating dtype used for internal arrays.
        """        
        # ---- store basic inputs ----
        self.r = float(r)
        self.A_s = float(A_s)
        self.N_pivot = float(N_pivot)
        self.N_max = float(N_max)
        self.N_min = float(N_min)
        self.k_pivot = float(k_pivot)
        self.dtype = dtype


        # ---- normalize arrays ----
        # eta: shape (P,)        
        self.eta = asarray_numeric(eta_list, dtype=self.dtype)
        # transitions: shape (P-1,)
        self.transitions = asarray_numeric(efold_list, dtype=self.dtype)

        P = self.eta.size
        if self.transitions.size != P - 1:
            raise ValueError(f"efold_list must have length P-1={P-1}, got {self.transitions.size}")

        # ---- build phase edges (descending) ----
        # edges_desc has shape (P+1,)
        # E[0]=N_max, E[P]=N_min, and transitions define the interior edges.
        self.edges_desc = np.concatenate(([self.N_max], self.transitions, [self.N_min])).astype(self.dtype)
        require_strictly_decreasing(self.edges_desc, name="edges_desc (N_max, efold_list, N_min)")

        # ---- sanity checks for pivot and pivot-scale parameters ----
        if not (self.N_min <= self.N_pivot <= self.N_max):
            raise ValueError(f"N_pivot={self.N_pivot} must lie within [{self.N_min}, {self.N_max}].")
                
        if self.k_pivot <= 0.0:
            raise ValueError("k_pivot must be positive.")

        # Ascending copy for searchsorted-based phase indexing
        self.edges_asc = self.edges_desc[::-1]

        # ---- pivot-derived attributes ----
        # H_pivot from (r, A_s) in reduced Planck units
        self.H_pivot = float(np.pi * np.sqrt(self.r * self.A_s / 2.0))
        self.epsilon_pivot = float(self.H_pivot**2 / (8.0 * np.pi**2 * self.A_s))

        # ---- caches (computed once) ----
        # _eps_edge[i] is epsilon at phase start edge E[i], for i=0..P
        self._eps_edge = self._build_epsilon_edge_cache()  # (P+1,)
        # _H_edge[i] is H at phase start edge E[i], for i=0..P
        self._H_edge = self._build_H_edge_cache()          # (P+1,)

        # Verify inflation condition at edges and raise if violated (conservative check)
        self.inflation_ok = self._check_inflation()

        # Check slow-roll hierarchy and warn if violated
        self.hierarchy_ok = self.check_slow_roll_hierarchy(warn=True)

    @property
    def n_phases(self):
        """Number of constant-eta phases (P)."""
        return int(self.eta.size)

    def phase_index(self, N):
        """
        Return the phase index for each N.

        Parameters
        ----------
        N : float or ndarray
            E-fold(s) before the end of inflation.

        Returns
        -------
        idx : ndarray of int64
            Phase indices with the same shape as N.
            Values are in [0, P-1].
        """
        N_arr = asarray_numeric(N, dtype=self.dtype)

        # bounds check        
        if np.any((N_arr > self.N_max) | (N_arr < self.N_min)):
            raise ValueError(f"N values are out of bounds [{self.N_min}, {self.N_max}]")

        # j indexes intervals in ascending-edge space: E[j] < N <= E[j+1]
        j = np.searchsorted(self.edges_asc, N_arr, side="left") - 1
        j = np.clip(j, 0, self.n_phases - 1)

        # convert to descending-phase indexing        
        idx = (self.n_phases - 1) - j
        return idx.astype(np.int64)

    def eta_H(self, N):
        """
        Evaluate eta_H(N) (piecewise constant) for given N.

        Parameters
        ----------
        N : float or ndarray

        Returns
        -------
        etaN : ndarray
            eta values at N, broadcasted to match N's shape.
        """
        idx = self.phase_index(N)
        return self.eta[idx]

    def epsilon_H(self, N):
        """
        Evaluate epsilon_H(N) for given N using cached phase-start values.

        Parameters
        ----------
        N : float or ndarray
            E-fold(s) before the end of inflation.

        Returns
        -------
        epsN : ndarray
            epsilon_H evaluated at N, same shape as N_arr.
        """
        N_arr = asarray_numeric(N, dtype=self.dtype)

        # identify phase for each N
        idx = self.phase_index(N_arr)

        # phase start edge and cached epsilon at that edge
        N_start = self.edges_desc[idx]
        eps_start = self._eps_edge[idx]
        eta = self.eta[idx]

        nz = ~isclose_zero(eta, atol=0.0, rtol=0.0)
        epsN = np.empty_like(N_arr, dtype=self.dtype)

        # eta != 0 branch
        epsN[nz] = (eps_start[nz] * eta[nz]) / (eps_start[nz] - (eps_start[nz] - eta[nz]) * np.exp(2.0 * eta[nz] * (N_start[nz] - N_arr[nz])))

        # eta == 0 branch
        epsN[~nz] = eps_start[~nz] / (1.0 + 2.0 * eps_start[~nz] * (N_arr[~nz] - N_start[~nz]))
        return epsN

    def Hubble(self, N):
        """
        Evaluate H(N) in units of reduced Planck mass for given N using cached phase-start values.

        Parameters
        ----------
        N : float or ndarray
            E-fold(s) before the end of inflation.

        Returns
        -------
        HN : ndarray
            Hubble parameter evaluated at N, same shape as N_arr.
        """
        N_arr = asarray_numeric(N, dtype=self.dtype)
        idx = self.phase_index(N_arr)

        # phase start edge and cached H, epsilon at that edge
        N_start = self.edges_desc[idx]
        eps_start = self._eps_edge[idx]
        H_start = self._H_edge[idx]
        eta = self.eta[idx]


        nz = ~isclose_zero(eta, atol=0.0, rtol=0.0)
        HN = np.empty_like(N_arr, dtype=self.dtype)

        # eta != 0 branch
        HN[nz]  = H_start[nz]  * (1 - (eps_start[nz] / eta[nz]) * (1 - np.exp(-2.0 *eta[nz] * (N_start[nz] - N_arr[nz]))))**(0.5)
        # eta == 0 branch
        HN[~nz] = H_start[~nz] * ( 1 + 2 * eps_start[~nz] * (N_arr[~nz] - N_start[~nz]) )** (0.5)
        return HN

    # ---- cache builders ----
    def _build_epsilon_edge_cache(self):
        """
        Build epsilon values at all phase edges.

        Returns
        -------
        eps_edge : ndarray, shape (P+1,)
            eps_edge[i] = epsilon at edge E[i] (phase start edge for phase i).
        """
        E = self.edges_desc
        eta = self.eta
        P = self.n_phases

        eps_edge = np.empty(P + 1, dtype=self.dtype)

        i_ref = int(self.phase_index(self.N_pivot))
        eta_ref = float(eta[i_ref])
        eps_ref = float(self.epsilon_pivot)
  
        # epsilon at the start edge of the pivot phase (inverse map from pivot to edge)
        if not isclose_zero(eta_ref, atol=0.0, rtol=0.0):
            eps_edge[i_ref] = eps_ref * eta_ref / (eps_ref - (eps_ref - eta_ref) *  np.exp(2.0 * eta_ref * (self.N_pivot - E[i_ref])))
        else:
            eps_edge[i_ref] = eps_ref / (1.0 + 2.0 * eps_ref * (E[i_ref] - self.N_pivot))    

        # propagate upward (toward larger N): compute eps at E[i] from eps at E[i+1]
        for i in range(i_ref -1, -1, -1):
            eta_i = float(eta[i])
            if not isclose_zero(eta_i, atol=0.0, rtol=0.0):
                eps_edge[i] = eps_edge[i + 1] * eta_i / (eps_edge[i + 1] - (eps_edge[i + 1] - eta_i) * np.exp(2.0 * eta_i * (E[i + 1] - E[i])))
            else:
                eps_edge[i] = eps_edge[i + 1] / (1.0 + 2.0 * eps_edge[i + 1] * (E[i] - E[i + 1]))

        # propagate downward (toward smaller N): compute eps at E[i+1] from eps at E[i]            
        for i in range(i_ref, P):
            eta_i = float(eta[i])
            if not isclose_zero(eta_i, atol=0.0, rtol=0.0):
                eps_edge[i + 1] = eps_edge[i] * eta_i / (eps_edge[i] - (eps_edge[i] - eta_i) * np.exp(2.0 * eta_i * (E[i] - E[i + 1])))
            else:
                eps_edge[i + 1] = eps_edge[i] / (1.0 + 2.0 * eps_edge[i] * (E[i + 1] - E[i]))


        return eps_edge

    def _build_H_edge_cache(self):
        """
        Build H values at all phase edges.

        Returns
        -------
        H_edge : ndarray, shape (P+1,)
            H_edge[i] = Hubble parameter at edge E[i] (phase start edge for phase i).
        """
        E = self.edges_desc
        eta = self.eta
        eps_edge = self._eps_edge
        P = self.n_phases

        H_edge = np.empty(P + 1, dtype=self.dtype)

        i_ref = int(self.phase_index(self.N_pivot))
        eta_ref = float(eta[i_ref])

        H_ref = float(self.H_pivot)
        eps_edge_ref = float(eps_edge[i_ref])

        # H at the start edge of the pivot phase (invert start->pivot factor)
        if not isclose_zero(eta_ref, atol=0.0, rtol=0.0):
            H_edge[i_ref] = H_ref * (1 - (eps_edge_ref / eta_ref) * (1 - np.exp(-2.0 * eta_ref * (E[i_ref] - self.N_pivot))))**(-0.5)
        else:
            H_edge[i_ref] = H_ref * (1.0 + 2.0 * eps_edge_ref * (self.N_pivot - E[i_ref]))**(-0.5)

        # propagate upward (toward larger N): invert start->end factor
        for i in range(i_ref - 1, -1, -1):
            eta_i = float(eta[i])
            if not isclose_zero(eta_i, atol=0.0, rtol=0.0):
                H_edge[i] = H_edge[i + 1] * (1 - (eps_edge[i] / eta_i) * (1 - np.exp(-2.0 * eta_i * (E[i] - E[i + 1]))))**(-0.5)
            else:
                H_edge[i] = H_edge[i + 1] * (1.0 + 2.0 * eps_edge[i] * (E[i + 1] - E[i]))** (-0.5)

        # propagate downward (toward smaller N): apply start->end factor
        for i in range(i_ref, P):
            eta_i = float(eta[i])
            if not isclose_zero(eta_i, atol=0.0, rtol=0.0):
                H_edge[i + 1] = H_edge[i] * (1 - (eps_edge[i] / eta_i) * (1 - np.exp(-2.0 * eta_i * (E[i] - E[i + 1]))))**(0.5)
            else:
                H_edge[i + 1] = H_edge[i] * (1.0 + 2.0 * eps_edge[i] * (E[i + 1] - E[i]))** (0.5)               

        return H_edge


    def check_slow_roll_hierarchy(self, *, warn=True):
        """
        Check the hierarchy epsilon_H < |eta_H| phase-by-phase.

        Parameters
        ----------
        warn : bool, default True
            If True, issue a RuntimeWarning if the hierarchy is violated in any phase, indicating the
            first failing phase index and the relevant eta_H and max-epsilon values.

        Returns
        -------
        ok : bool
            True if satisfied in all phases, False otherwise.
        """
        eta = self.eta
        eps_edge = self._eps_edge
        P = self.n_phases

        max_eps = np.maximum(eps_edge[:P], eps_edge[1:P+1])
        violated = ~(max_eps < np.abs(eta))
        ok = not np.any(violated)

        if (not ok) and warn:
            i0 = int(np.where(violated)[0][0])
            warnings.warn(
                "Slow-roll hierarchy violated (epsilon_H < |eta_H|). "
                "Background functions remain usable, but Spectrum approximations may be invalid. "
                f"First failing phase index: {i0}, eta_H={eta[i0]}, max_eps={max_eps[i0]}",
                RuntimeWarning,
            )
        return ok
    
    def _check_inflation(self, *, tol=0.0):
        """
        Ensure inflation condition epsilon_H < 1 holds throughout all phases.

        Uses the cached edge values as a conservative check.
        Raises ValueError if epsilon reaches or exceeds 1.
        """
        eps_edge = self._eps_edge
        P = self.n_phases
        max_eps = np.maximum(eps_edge[:P], eps_edge[1:P+1])

        if np.any(max_eps >= 1.0 - tol):
            i0 = int(np.where(max_eps >= 1.0 - tol)[0][0])
            raise ValueError(
                "Inflation condition violated: epsilon_H >= 1 within the defined history.\n"
                f"First failing phase index: {i0}\n"
                f"max epsilon in phase = {max_eps[i0]}"
            )
        
        return True
        
    def summary(self, *, N_end=None, digits=6) -> str:
        """
        Return a human-readable summary of the defined inflationary history.

        Parameters
        ----------
        N_end : float, optional
            N value near the end to report epsilon/H at. Defaults to N_min.
        digits : int, default 6
            Significant digits for formatting.

        Returns
        -------
        s : str
            Multi-line summary string.
        """
        if N_end is None:
            N_end = self.N_min

        # Pivot quantities
        eta_p = float(self.eta_H(self.N_pivot))
        eps_p = float(self.epsilon_pivot)


        # n_s at pivot (your stated approximation)
        n_s = 1.0 + 2.0 * eta_p - 4.0 * eps_p

        # Near end quantities
        eta_e = float(self.eta_H(N_end))
        eps_e = float(self.epsilon_H(N_end))
        H_e = float(self.Hubble(N_end))

        # Optional flags if present
        hierarchy_ok = getattr(self, "hierarchy_ok", None)

        fmt = f"{{:.{digits}g}}"
        def f(x):  # short formatter
            return fmt.format(x)

        lines = []
        lines.append("InflationHistory summary")
        lines.append("-" * 72)
        lines.append("Inputs")
        lines.append(f"  r           = {f(self.r)}")
        lines.append(f"  A_s         = {f(self.A_s)}")
        lines.append(f"  N_pivot     = {f(self.N_pivot)}")
        lines.append(f"  k_pivot     = {f(self.k_pivot)}  [Mpc^-1]")
        lines.append(f"  N_range     = [{f(self.N_min)}, {f(self.N_max)}]  (before-end convention)")
        lines.append("")
        lines.append("Phases")
        lines.append(f"  n_phases    = {self.n_phases}")
        lines.append(f"  eta_list    = {np.array2string(self.eta, precision=digits)}")
        lines.append(f"  efold_list  = {np.array2string(self.transitions, precision=digits)}")
        lines.append("")
        lines.append("Pivot-derived")
        lines.append(f"  H_pivot     = {f(self.H_pivot)} [m_Planck]")
        lines.append(f"  epsilon_pivot   = {f(self.epsilon_pivot)}")
        lines.append(f"  eta_pivot   = {f(eta_p)}")
        lines.append(f"  n_s(pivot)  = {f(n_s)}    (1 + 2η - 4ε)")
        lines.append("")
        lines.append(f"Near end (N = {f(N_end)})")
        lines.append(f"  eta(end)    = {f(eta_e)}")
        lines.append(f"  epsilon(end)    = {f(eps_e)}")
        lines.append(f"  H(end)      = {f(H_e)} [m_Planck]")
        lines.append("")
        lines.append("Validity flags")
        if hierarchy_ok is not None:
            lines.append(f"  hierarchy_ok (eps < |eta|) = {hierarchy_ok}")
        else:
            lines.append("  hierarchy_ok (eps < |eta|) = (not computed)")
        # If you add an inflation check attribute, include it similarly:
        infl_ok = getattr(self, "inflation_ok", None)
        if infl_ok is not None:
            lines.append(f"  inflation_ok (eps < 1)     = {infl_ok}")
        lines.append("-" * 72)

        return "\n".join(lines)
        



