# Brain Tumor Detection using Deep Learning 
### This project focuses on building a deep learning model to classify brain MRI images into categories such as glioma, meningioma, pituitary tumor, and no tumor. The model is trained using a CNN / ResNet architecture on a publicly available dataset.

## Features
-- Automatic dataset download from Kaggle

-- Data preprocessing and augmentation

-- Training & validation pipelines

-- Model checkpointing and evaluation

-- High accuracy using CNN / ResNet backbone

-- Easy to extend for other medical image datasets
## Downloading the Dataset

To download the Brain Tumor Dataset:

1. **Install the Kaggle API**:
   ```bash
   pip install kaggle

2. **Setup kaggle API credentials**:
   Refer to Kaggle API documentations and download the kaggle.json file.
   Place kaggle.json in your home directory (~/.kaggle/)

3. **Add the Script to Your GitHub Repository**

- Place the `download_dataset.py` script in your repository.
- Commit and push the changes:
  ```bash
  git add download_dataset.py README.md
  git commit -m "Add dataset download script and update README"
  git push origin main
