# WASP_Python

An open-source Python port of the Wavelet System Prediction (WASP) tool developed by Dr. Ze Jiang for improving prediction accuracy in natural system models.

This version replicates the core functionality of the original [WASP_MATLAB](https://github.com/zejiang-unsw/WASP_matlab) package, using Python and PyWavelets, enabling full calibration and validation workflows with wavelet-domain variance transformation.

---

## 💡 What is WASP?

WASP (Wavelet System Prediction) is a method to refine predictor spectral representations using frequency-domain variance transformation. It improves model accuracy by aligning predictor variance with the spectral structure of the target response, commonly applied in hydrological and environmental forecasting.

This implementation uses:
- Custom-built multiresolution decomposition (`dwt_mra`)
- Wavelet-based covariance weighting (`wasp`)
- Forecast validation with precomputed weights (`wasp_val`)

---

## 🔧 Requirements

Install the required Python packages using pip:

```bash
pip install numpy matplotlib PyWavelets
```

## Citation
Jiang, Z., Sharma, A., & Johnson, F. (2020).
Refining Predictor Spectral Representation Using Wavelet Theory for Improved Natural System Modeling.
Water Resources Research, 56(3), e2019WR026962.
https://doi.org/10.1029/2019WR026962

Jiang, Z., Rashid, M. M., Johnson, F., & Sharma, A. (2020).
A wavelet-based tool to modulate variance in predictors: An application to predicting drought anomalies.
Environmental Modelling & Software, 135, 104907.
https://doi.org/10.1016/j.envsoft.2020.104907

Jiang, Z., Sharma, A., & Johnson, F. (2021).
Variable transformations in the spectral domain – Implications for hydrologic forecasting.
Journal of Hydrology, 603, 126816.
https://doi.org/10.1016/J.JHYDROL.2021.126816


