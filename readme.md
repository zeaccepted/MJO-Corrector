# MJO-Corrector

This repository contains the code used for "A Physics-Guided AI Post-processor Significantly Extends Madden-Julian Oscillation Prediction Skill".


## 📖 Introduction
The Madden-Julian Oscillation (MJO) is an important driver of global weather and climate extremes, but its prediction in operational dynamical models remains challenging, with skillful forecasts typically limited to 3-4 weeks. Here, we introduce a novel deep learning framework, the Physics-guided Cascaded Corrector for MJO (PCC-MJO), which acts as a universal post-processor to correct MJO forecasts from dynamical models. This two-stage model first employs a physics-informed 3D U-Net to correct spatial-temporal field errors, then refines the MJO’s RMM index using an LSTM optimized for forecast skill. When applied to three different operational forecasts from CMA, ECMWF and NCEP, our unified framework consistently extends the skillful forecast range (bivariate correlation > 0.5) by 2-8 days. Crucially, the model effectively mitigates the "Maritime Continent barrier," enabling more realistic eastward propagation and amplitude. Explainable AI analysis quantitatively confirms that the model's decision-making is spatially congruent with observed MJO dynamics (correlation > 0.93), demonstrating that it learns physically meaningful features rather than statistical fittings. Our work provides a promising physically consistent, computationally efficient, and highly generalizable pathway to break through longstanding barriers in subseasonal forecasting.

<img width="5016" height="1601" alt="mjo-corrector框架" src="https://github.com/user-attachments/assets/0e57c710-faf9-4f1d-8518-618836543b54" />




### Prerequisites
- Python 3.8+
- PyTorch 1.12+

## 📁 Dataset
The atmospheric circulation data used in this study are from the NCEP/DOE Reanalysis-2 (R2) dataset, accessible through the Physical Sciences Laboratory (PSL) at NOAA: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.pressure.html. The outgoing longwave radiation (OLR) data are obtained from the NOAA Interpolated OLR product, available at: https://psl.noaa.gov/data/gridded/data.olrcdr.interp.html. The subseasonal forecast data were provided by the respective modeling centers (ECMWF, BCC, and NCEP) and can be acquired through their institutional data portals or upon request.

## 🏃‍♂️ Training and Evaluation
1. Training
```bash
python train.py
```
2. Evaluation
```bash
python test.py
```
## 📜 Citation


## 📧 Contact
For any questions or suggestions, please contact [Yuze Sun] at [syz23@mails.tsinghua.edu.cn] or open an issue on GitHub.

## 📄 License
This project is licensed under the MIT License.
