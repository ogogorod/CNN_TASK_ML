# Weapon Classification with CNN and Transfer Learning

This project implements an image-classification pipeline for visible object and weapon detection.  
The task is a 6-class classification problem with an additional binary weapon-detection interpretation.
## Dataset
Original dataset: [OD-WeaponDetection: Sohas Detection Dataset](https://datasetninja.com/od-weapon-detection-sohas-detection)

Check points for models: [here](https://drive.google.com/file/d/1Dnjoz_YI_TFjakZOJSrq3upEHP213jnD/view?usp=sharing)

The original dataset used in this project is the OD-WeaponDetection / Sohas Detection Dataset, available from DatasetNinja at the link above. The dataset contains images and annotation files used for multiclass object classification and binary weapon/non-weapon detection. Because the full dataset and trained model checkpoint files are too large to store directly in this GitHub repository, the processed project data and trained checkpoints are provided separately through the Google Drive link. To reproduce the project, download the files from Google Drive and place them in the expected folders described below.

## Project Goal

The goal is to compare a custom CNN against transfer-learning models for classifying images into:

- `billete`
- `knife`
- `monedero`
- `pistol`
- `smartphone`
- `tarjeta`

For the binary weapon-detection task:

- weapon classes: `knife`, `pistol`
- non-weapon classes: `billete`, `monedero`, `smartphone`, `tarjeta`

The project evaluates both multiclass classification performance and binary weapon/non-weapon detection performance.

## Project Structure

Expected folder structure:

```text
weapon_classifier/
    README.md
    requirements.txt

    data/
        raw/
            img/
            ann/

        test/
            img/
            ann/
            labels.csv
            summary.txt

    outputs/
        checkpoints/
            scratch_best.pt
            transfer_best.pt
            mobilenetv2_best.pt

        reports/
            scratch_history.json
            transfer_history.json
            mobilenetv2_history.json

        predictions/

        figures/
            confusion_matrices/
            roc_curves/
            precision_recall_curves/
            training_curves/
            dataset_summary/

        gradcam/

    src/
        config.py
        dataset.py
        transforms.py
        models_v2.py
        train_v2.py
        infer.py
        evaluate.py
        export_test_dataset.py
        gradcam.py
        plot_training_curves.py
        dataset_summary.py
        utils.py
```

## Dataset Layout

The raw dataset should be placed here:

```text
data/raw/img/
data/raw/ann/
```

The expected annotation format is:

```text
image: data/raw/img/example.jpg
annotation: data/raw/ann/example.jpg.json
```

Labels are extracted from annotation JSON files. If a label cannot be read from JSON, the code falls back to the first filename token before `_`.

## Models

The project supports three models:

1. **Scratch CNN**
   - Custom convolutional neural network trained from random initialization.

2. **Transfer model**
   - ResNet18 pretrained on ImageNet and adapted to 6 output classes.

3. **MobileNetV2**
   - MobileNetV2 pretrained on ImageNet and adapted to 6 output classes.

All models output 6 logits corresponding to the fixed class order in `config.py`.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended dependencies include:

```text
torch>=2.1
torchvision>=0.16
scikit-learn>=1.3
matplotlib>=3.7
Pillow>=10.0
numpy>=1.24
certifi>=2024.0.0
pandas>=3.0.3
opencv-python>=4.8.0
```

## Configuration

All important paths and hyperparameters are defined in:

```text
src/config.py
```

The project root is detected automatically from the location of `config.py`, so scripts can be run from the project root with commands like:

```bash
python src/evaluate.py
```

Key paths:

```text
data/raw/              raw dataset
data/test/             exported balanced test dataset
outputs/checkpoints/   saved model weights
outputs/reports/       metrics and CSV/JSON summaries
outputs/figures/       generated plots
outputs/gradcam/       Grad-CAM visualizations
```

## Training

Train one model:

```bash
python src/train_v2.py --model scratch
python src/train_v2.py --model transfer
python src/train_v2.py --model mobilenetv2
```

The training script has overwrite protection. If a checkpoint already exists, it will not overwrite it unless explicitly requested:

```bash
python src/train_v2.py --model mobilenetv2 --overwrite
```

Saved checkpoints:

```text
outputs/checkpoints/scratch_best.pt
outputs/checkpoints/transfer_best.pt
outputs/checkpoints/mobilenetv2_best.pt
```

Saved training histories:

```text
outputs/reports/scratch_history.json
outputs/reports/transfer_history.json
outputs/reports/mobilenetv2_history.json
```

## Export Balanced Test Dataset

Create a balanced evaluation dataset from the official held-out test split:

```bash
python src/export_test_dataset.py
```

This creates:

```text
data/test/img/
data/test/ann/
data/test/labels.csv
data/test/summary.txt
```

To recreate the folder if it already exists:

```bash
python src/export_test_dataset.py --overwrite
```

Optional fixed number of samples per class:

```bash
python src/export_test_dataset.py --samples-per-class 50
```

## Inference

Run inference on one image:

```bash
python src/infer.py --model scratch --image data/test/img/example.jpg
python src/infer.py --model transfer --image data/test/img/example.jpg
python src/infer.py --model mobilenetv2 --image data/test/img/example.jpg
```

Run inference on a folder or glob pattern:

```bash
python src/infer.py --model mobilenetv2 --image "data/test/img/*.jpg"
```

## Evaluation

Evaluate all models on the exported balanced test dataset:

```bash
python src/evaluate.py
```

Quick test on a small subset:

```bash
python src/evaluate.py --limit 30
```

Evaluate selected models:

```bash
python src/evaluate.py --models scratch transfer
```

The evaluation script calculates:

- accuracy
- balanced accuracy
- precision
- recall
- F1-score
- sklearn-style classification report
- multiclass confusion matrix
- binary weapon/non-weapon confusion matrix
- binary ROC-AUC
- binary precision-recall AUC
- false positives
- false negatives
- inference time per image
- images per second
- model size
- parameter count

Main outputs:

```text
outputs/reports/evaluation_results.json
outputs/reports/model_comparison.csv
outputs/reports/efficiency_metrics.csv

outputs/predictions/predictions_<model>.csv
outputs/predictions/errors_<model>.csv
outputs/predictions/false_positives_<model>.csv
outputs/predictions/false_negatives_<model>.csv

outputs/figures/confusion_matrices/
outputs/figures/roc_curves/
outputs/figures/precision_recall_curves/
```

## Training Curves

Generate training and validation accuracy/loss curves:

```bash
python src/plot_training_curves.py
```

Outputs:

```text
outputs/figures/training_curves/
outputs/reports/training_curve_summary.csv
```

## Dataset Summary

Generate class-distribution summaries for the raw dataset, recreated splits, and exported balanced test dataset:

```bash
python src/dataset_summary.py
```

Outputs:

```text
outputs/reports/dataset_summary.csv
outputs/reports/dataset_summary.json

outputs/figures/dataset_summary/raw_class_distribution.png
outputs/figures/dataset_summary/train_val_test_distribution.png
outputs/figures/dataset_summary/exported_test_class_distribution.png
```

## Grad-CAM Explainability

Generate Grad-CAM visualizations after running evaluation:

```bash
python src/gradcam.py
```

Recommended for the report:

```bash
python src/gradcam.py --num-images 3
```

Outputs:

```text
outputs/gradcam/
    scratch/
        correct/
        false_positives/
        false_negatives/

    transfer/
        correct/
        false_positives/
        false_negatives/

    mobilenetv2/
        correct/
        false_positives/
        false_negatives/
```

Grad-CAM examples are generated for:

1. correct predictions
2. false positives
3. false negatives

If no binary false positives or false negatives exist for a model, the script falls back to multiclass errors.

## Recommended Full Pipeline

Run these commands from the project root:

```bash
python src/export_test_dataset.py
python src/evaluate.py
python src/plot_training_curves.py
python src/dataset_summary.py
python src/gradcam.py --num-images 3
```

If the test dataset already exists and you want to recreate it:

```bash
python src/export_test_dataset.py --overwrite
```



## Notes on Reproducibility

The train/validation/test split is controlled by:

```text
config.SEED = 42
```

The exported balanced test dataset is created from the official held-out test split using the same seed and split logic as the training pipeline.

## Ethical and Practical Limitations

This project is an academic prototype. It should not be used as a real security or law-enforcement system. Potential limitations include:

- dataset bias
- limited class coverage
- false positives on harmless objects
- false negatives on weapons
- sensitivity to background, lighting, occlusion, and image quality
- lack of real-world deployment testing

The model outputs should be interpreted as experimental classification results, not as automated security decisions.
