# Changelog

All notable changes to **ZetaInSpect** will be documented in this file.

The format is based on *Keep a Changelog*, and this project follows *Semantic Versioning*.

## [0.1.0] - 2026-02-23

### Added
- `InflationHistory`: piecewise-constant η_H inflationary background with cached edge values for ε_H(N) and H(N).
- Inflation validity check: raises if ε_H reaches or exceeds 1 (conservative edge-based check).
- Slow-roll hierarchy check: `epsilon_H < |eta_H|` check recorded as a boolean and emitted as a warning when violated.
- `Scales`: conversions between N, comoving horizon-exit scale k_exit(N), evaluation scale k(N, ratio), and x=k/(aH).
- `Spectrum`: Mukhanov–Sasaki Hankel matching solution for:
  - ratio-mode spectrum `vk(N, ratio)` and `power_spectrum(N, ratio)`
  - fixed-k tracking `vk_track(N, k)` and `power_spectrum_track(N, k)`
- Performance improvements:
  - Vectorized, masked coefficient updates across transitions (no Python loops over N grids).
  - Fixed-k coefficient caching to avoid recomputation across repeated calls.
  - Phase-wise evaluation for tracking to minimize unnecessary Hankel evaluations.
- `plotting`: dispatcher to plot η, |η|, ε, H, v_k, and P_ζ versus N or k, with optional overlays (pivot, transitions, A_s, CMB window, PBH threshold).

### Changed
- None (initial public release).

### Fixed
- None (initial public release).

### Notes
- SciPy (`scipy.special.hankel1/2`) is used for Hankel functions; GPU backends are not enabled for these special functions.
- Plotting is an optional extra (`zetainspect[plot]`) to keep non-plot workflows lightweight.