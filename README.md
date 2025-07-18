# EarthESND-Model

## 🧩 EarthESND Model

**EarthESND** is a multi-scale, multi-layer Echo State Network (ESN) designed for early earthquake magnitude prediction.  
The main pipeline is implemented in the `EarthESND.ipynb` notebook, which uses a modular architecture.  
Core ESN functionalities are imported from the `MultiScaleESN.py` module, which implements the custom multi-scale, multi-layer ESN structure.

- **Main file:** `EarthESND.ipynb`  
  Runs the full experiment, including training, and evaluation.

- **ESN module:** `MultiScaleESN.py`  
  Contains the custom class definitions for the multi-scale, multi-layer ESN.

The network processes both waveform features and tabular seismic features to provide robust, real-time earthquake magnitude estimates.

---

### 📌 **Input Parameters**

#### ✅ **1️⃣ Waveform Features**

| Parameter | Symbol | Description |
|-------------------------------|------------------------|------------------------------------------|
| North-South Acceleration      | $\mathcal{A}_{NS}$     | NS component of acceleration waveform |
| East-West Acceleration        | $\mathcal{A}_{EW}$     | EW component of acceleration waveform |
| Up-Down Acceleration          | $\mathcal{A}_{UD}$     | UD (vertical) component of acceleration waveform |
| North-South Velocity          | $\mathcal{V}_{NS}$     | NS component of velocity waveform |
| East-West Velocity            | $\mathcal{V}_{EW}$     | EW component of velocity waveform |
| Up-Down Velocity              | $\mathcal{V}_{UD}$     | UD (vertical) component of velocity waveform |
| North-South Displacement      | $\mathcal{D}_{NS}$     | NS component of displacement waveform |
| East-West Displacement        | $\mathcal{D}_{EW}$     | EW component of displacement waveform |
| Up-Down Displacement          | $\mathcal{D}_{UD}$     | UD (vertical) component of displacement waveform |

---

#### ✅ **2️⃣ Tabular Features**

| Parameter | Symbol | Description |
|--------------------------------------------|----------------|----------------------------------------------------------|
| Characteristic Period                      | $\tau_c$       | Time period characterizing the source signal |
| Integrated Squared Displacement            | $ID2$          | Integral of squared displacement over time |
| Integrated Squared Velocity                | $IV2$          | Integral of squared velocity over time |
| Peak Integrated Velocity                   | $PI_V$         | Peak value of integrated velocity |
| Root Sum of Squared Velocity               | $RSSCV$        | Square root of the sum of squared velocity components |
| Peak Velocity Acceleration Ratio           | $T_{va}$       | Ratio of peak velocity to acceleration |
| Cumulative Absolute Velocity               | $CAAV$         | Cumulative sum of absolute velocity over time |

---

**Together**, these waveform and tabular features are used for early earthquake magnitude prediction.
