# BarleyYield-3DPointCloud-ML

## 📌 Overview
This repository contains the code, workflow, and data processing pipeline for a UAV-based **high-throughput phenotyping** method designed to predict barley yield under varying nitrogen conditions.  
We integrate **RGB-multispectral sensors** with high-density **3D point cloud reconstruction** to extract growth stage-specific spectral indices and structural traits, enabling **non-destructive, rapid, and multi-site synchronized yield estimation**.

---

## ✨ Key Features
- **Novel Preprocessing Workflow** — Balances model accuracy with growth stage-specific spectral indicators.
- **3D Spectral Indices** — Overcome 2D orthomosaic limitations by capturing plant height, biomass, and canopy structure.
- **Higher Prediction Accuracy** — R² improved by 18–28%, RMSE reduced by 15–22% over classical vegetation indices.
- **Low-Nitrogen Robustness** — Accurately distinguishes high- and low-yield genotypes under nitrogen stress.
- **Cross-Experiment Scalability** — Works across regions and stress conditions.

---

## 🛠 Methodology
1. **Data Acquisition**
   - UAV flights with integrated RGB & multispectral sensors.
   - Simultaneous high-density 3D point cloud generation.
2. **Feature Extraction**
   - Derive spectral indices from 3D point cloud data.
   - Incorporate plant height, biomass, and canopy architecture.
3. **Preprocessing**
   - Stage-specific filtering and index selection.
4. **Machine Learning Modeling**
   - Yield prediction models trained and validated across sites.
5. **Performance Evaluation**
   - Compare with classical 2D vegetation indices.

---

## 📂 Repository Structure
