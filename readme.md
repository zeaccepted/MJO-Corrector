# MJO-Corrector

This repository contains the code used for "A Physics-Guided AI Post-processor Significantly Extends Madden-Julian Oscillation Prediction Skill".


## 📖 Introduction



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
