# 🏠 Multimodal ML – Housing Price Prediction (Images + Tabular Data)

## 📌 Objective
The goal of this project is to **predict housing prices** using both structured **tabular data** and **house images**.  
We leverage **Convolutional Neural Networks (CNNs)** for extracting image features and combine them with structured features to build a **multimodal regression model**.

---

## 📂 Dataset
The dataset consists of:
- **Tabular Data**: Housing attributes such as size, number of rooms, location, etc.  
- **Image Data**: Pictures of houses corresponding to the tabular entries.  

Each house has both **structured features** and **an image**.

---

## ⚙️ Methodology
1. **Preprocess Tabular Data**  
   - Handle missing values  
   - Normalize/scale numerical features  
   - Encode categorical features  

2. **Image Feature Extraction**  
   - Resize images to a fixed size (e.g., 224×224)  
   - Use pretrained CNN (ResNet50, VGG16, etc.) to extract embeddings  

3. **Feature Fusion**  
   - Concatenate CNN-based embeddings with tabular features  
   - Train dense layers for regression  

4. **Evaluation**  
   - **Loss function:** Mean Squared Error (MSE)  
   - **Metrics:** MAE, RMSE  

---

## 🚀 Training
Run the training script:

```bash
python train.py
