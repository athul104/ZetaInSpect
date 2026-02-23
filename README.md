# ZetaInSpect

**Scalar Curvature Power Spectrum Inflationary Spectra**

[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-brightgreen)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)

---

**ZetaInSpect** is a lightweight, research-grade Python package to compute and inspect:

- The inflationary background history with **piecewise-constant** Hubble slow-roll parameter \(\eta_H\).
- The Mukhanov–Sasaki mode function \(v_k\) using Hankel matching across phases.
- The primordial curvature power spectrum \(P_\zeta(k)\) (or \(P_\zeta(N)\)) in a convenient, vectorized workflow.

Plotting utilities are included as an optional extra.

---

## 📦 Installation

### 🧪 From GitHub (temporary)

```bash
pip install git+https://github.com/athul104/ZetaInSpect.git
```

### 🛠️ Cloning repo to your local system

```bash
git clone https://github.com/athul104/ZetaInSpect.git
cd ZetaInSpect
pip install -e .
```

### Optional plotting dependencies

```bash
pip install -e ".[plot]"
```

## 📋 Requirements
* Python **3.10** or later
* ```numpy >= 1.23```
* ```scipy >= 1.9```
* ```matplotlib >= 3.7``` (optional, for plotting)



## 📓 Tutorial Notebook

A tutorial notebook will be placed under:

```bash
tutorials/examples.ipynb
```

## 📚 Citation

This package accompanies the paper:

> **Swagat S. Mishra, Sanket Dave, Athul K. Soman & Parth Bhargava (2026)**  
> *paper name*  
> [arXiv:XXXX.YYYYY](https://arxiv.org/abs/XXXX.YYYYY)

If you use ZetaInspect in academic work, please cite:

```bibtex

```
---

## 🛠 License
GPL-3.0-or-later  
©️ ZetaInspect authors (2026)


## 🙋 Feedback

Thank you for your interest in **ZetaInSpect** 💜.  
Please report issues or suggestions at:
[github.com/athul104/ZetaInSpect/issues](https://github.com/athul104/ZetaInSpect/issues)

---
