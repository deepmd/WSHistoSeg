# Weakly Supervised Histology Segmentation

# Training the Model with Weakly-Supervised Settings

To train the model using weakly-supervised settings, run the following command:

```bash
bash scripts/train_contrastive_selftraining.sh
```

Ensure that the images from the **GLAS dataset** are available at the following path:

```
GLAS/Warwick_QU_Dataset_(Released_2016_07_08)
```

Additionally, create a directory at the following path and place the CAMs (Class Activation Maps) generated using the **Grad-CAM** approach in it:

```
GLAS/Warwick_QU_Dataset_(Released_2016_07_08)/CAMs/training_cams
```