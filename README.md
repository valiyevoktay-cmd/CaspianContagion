# Caspian Contagion
> **Decoding Market Toxicity through Self-Exciting Processes**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-Hawkes_Process-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/UI-WebGL_Optimized-FF4B4B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

> An analytical engine that detects **market toxicity** and **self-exciting order flow contagion** in real-time cryptocurrency markets. Streams Binance L2 WebSocket data, computes Order Flow Imbalance, calibrates a continuous Hawkes Process via MLE, and triggers a protective **Killswitch** when the Contagion Condition Index signals adverse selection risk.

---

## 🔬 Academic Foundation

Caspian Contagion quantifies two specific microstructural phenomena:

1. **Order Flow Imbalance (OFI):** Vectorized multi-level computation measuring the net pressure of aggressive participants against resting limit orders — beyond naive volume aggregation.
2. **Hawkes Process (Contagion Condition Index):** Markets under informed flow exhibit self-exciting dynamics. The Hawkes model quantifies this via the **CCI = α/β**. When CCI ≥ 1.0, the market is in an endogenous adverse selection feedback loop — a mathematically confirmed toxic state for passive market-making.

---

## 🧮 Mathematical Framework

### Hawkes Process Intensity

The conditional intensity function of the Hawkes process:

$$\lambda(t) = \mu + \alpha \sum_{t_i < t} e^{-\beta(t - t_i)}$$

| Parameter | Meaning |
| :--- | :--- |
| $\mu$ | Baseline arrival rate (exogenous flow) |
| $\alpha$ | Excitation magnitude (shock size) |
| $\beta$ | Decay rate (mean-reversion speed) |
| $\alpha/\beta$ | **Contagion Condition Index (CCI)** — branching ratio |

### Recursive Log-Likelihood (O(N) computation)

$$R(i) = e^{-\beta \Delta t}(1 + R(i-1))$$

Parameters $(\mu, \alpha, \beta)$ calibrated via **L-BFGS-B** optimizer (`scipy.optimize.minimize`).

### Killswitch Trigger

$$\text{KILLSWITCH} \iff CCI = \frac{\alpha}{\beta} \geq 1.0$$

---

## 🏗 Core Architecture

```
CaspianContagion/
├── src/
│   ├── config.py              # Global constants & pipeline settings
│   ├── module_1_ingestion.py  # Asyncio Binance L2 WebSocket ingester
│   ├── module_2_hawkes.py     # Hawkes MLE Calibrator (L-BFGS-B, O(N) recursive kernel)
│   ├── module_3_dynamics.py   # OFI & microstructure calculator
│   ├── module_4_risk.py       # CCI computation & Killswitch engine
│   └── module_5_execution.py  # Naive vs contagion-aware execution simulator
├── ui/
│   └── dashboard.py           # Zero-flicker Streamlit HFT terminal (WebGL Plotly)
├── main.py                    # Pipeline launcher & background thread bridge
└── requirements.txt
```

**Key engineering decisions:**
- **Thread-safe global bridge** separating WebSocket ingestion from Streamlit rendering
- **Zero-flicker UI** via `uirevision='constant'` on Plotly — eliminates DOM re-render tearing at 1Hz refresh
- **O(N) recursive kernel** avoids $O(N^2)$ naive double-sum for Hawkes log-likelihood

---

## ⚙️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| Language | Python 3.10+ |
| Concurrency | `asyncio`, `threading` |
| Quantitative | `numpy`, `pandas`, `scipy` (L-BFGS-B) |
| Visualization | `plotly` (WebGL), `streamlit` |
| Data Source | Binance WebSocket `@depth20@100ms` |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/valiyevoktay-cmd/CaspianContagion.git
cd CaspianContagion

# 2. Create isolated environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch HFT terminal
streamlit run main.py
```

> **Note:** Allow 30–60 seconds for the initial warm-up phase. The system builds a localized LOB baseline before bridging to the live rendering engine.

---

## 🗺 Research Roadmap

- [ ] **Markout Analytics:** Post-trade execution quality metrics (price movement at T+100ms / T+500ms) to empirically validate Killswitch alpha
- [ ] **LOB Heatmaps:** 2D-density shaders to visualize hidden liquidity walls and spoofing patterns prior to Hawkes-triggered events
- [ ] **Dynamic Slippage Models:** Non-linear order book depth functions to convert `POTENTIAL_SLIPPAGE` into real-time basis points
- [ ] **Unit Test Suite:** Deterministic pytest coverage for Hawkes MLE calibration and OFI computation

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Disclaimer: This software is provided for research and academic purposes only. It does not constitute financial advice.*
