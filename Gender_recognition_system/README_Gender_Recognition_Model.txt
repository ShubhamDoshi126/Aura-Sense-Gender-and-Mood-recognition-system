# Gender Recognition Model

## Overview
This project implements a **Gender Recognition Model** using deep learning. The model is trained on the CelebA dataset to classify images as either **Male** or **Female**.

## Technologies Used
- **Python**
- **TensorFlow/Keras**
- **MobileNetV2** (pre-trained base model for transfer learning)
- **Pandas** for data preprocessing
- **Scikit-learn** for dataset splitting and evaluation

## Dataset
- **CelebA Dataset**:
  - Images: `img_align_celeba`
  - Labels: `list_attr_celeba.csv`
  - Gender mapping: `Male` → 1, `Female` → 0

## Steps Performed
1. **Data Preprocessing**:
   - Filter valid images.
   - Map gender labels (`-1` to 0 for Female, `1` for Male).
   - Split into train, validation, and test datasets.

2. **Model Development**:
   - **Transfer Learning**: MobileNetV2 as the feature extractor.
   - **Custom Layers**: Added dense layers with Dropout for regularization.

3. **Training and Fine-Tuning**:
   - Train initial layers.
   - Fine-tune the base model with a reduced learning rate.

4. **Evaluation**:
   - Evaluate on the test dataset and compute accuracy.

5. **Model Saving**:
   - Save the trained model as `fine_tuned_gender_recognition_model.h5`.

## Instructions to Run
1. Clone the repository and set up dependencies:
   ```bash
   pip install tensorflow pandas scikit-learn
   ```

2. Ensure the dataset is downloaded and paths are updated in the code.

3. Run the script:
   ```bash
   python Gender_Recognition_Model.py
   ```

4. The trained model will be saved as `gender_recognition_model.h5`.

## Results
- Test accuracy achieved: 98.15%.

## Future Improvements
- Implement data augmentation for improved generalization.
- Use additional datasets for more robust training.
