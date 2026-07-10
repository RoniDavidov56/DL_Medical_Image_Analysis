# Comparative Evaluation of U-Net-Based Architectures for Time-Lapse Cell Segmentation

This repository compares three U-Net-based deep-learning configurations for cell segmentation in time-lapse fluorescence microscopy. The study uses a common sequence-segmentation framework and evaluates how attention gates and residual blocks affect spatial feature extraction, boundary preservation, and cell detection.

The project evaluates the following configurations under the same dataset and training conditions:

1. **LSTM-U-Net** — the baseline architecture.
2. **LSTM-U-Net with Attention** — adds spatial attention gates to the skip connections.
3. **LSTM-ResU-Net** — replaces standard convolutional blocks with residual blocks.

## Project Objective

Cell segmentation in time-lapse microscopy is challenging because cells may overlap, appear with low contrast, change shape, enter or leave the field of view, or have weak and unclear boundaries.

Processing each frame independently can lead to inconsistent segmentation across time. The ConvLSTM component addresses this limitation by using information from consecutive frames, while the U-Net component extracts spatial features and reconstructs the segmentation masks.

The purpose of the project is to examine whether upgrading the spatial component of the baseline LSTM-U-Net improves cell detection, boundary preservation, and instance segmentation quality.

## Repository Files

```text
.
├── lstm_u_net_baseline.py
├── lstm_u_net_+_attention.py
├── lstm_resunet.py
└── README.md
```

| File | Architecture | Main modification |
|---|---|---|
| `lstm_u_net_baseline.py` | LSTM-U-Net | Baseline encoder-decoder with ConvLSTM layers and standard skip connections |
| `lstm_u_net_+_attention.py` | LSTM-U-Net with Attention | Spatial attention gates filter encoder features before decoder fusion |
| `lstm_resunet.py` | LSTM-ResU-Net | Standard convolutional blocks are replaced with residual blocks |

## Architectures

### 1. LSTM-U-Net Baseline

The baseline follows a U-shaped encoder-decoder architecture with skip connections and ConvLSTM-based temporal feature extraction.

The encoder extracts multi-scale spatial features from each frame. ConvLSTM layers model temporal relationships across the sequence, and the decoder reconstructs a segmentation map for every frame.

Main components:

- multi-level encoder-decoder structure;
- ConvLSTM layers for temporal feature extraction;
- encoder-decoder skip connections;
- batch normalization;
- Leaky ReLU activations;
- bilinear upsampling;
- three-class softmax output.

### 2. LSTM-U-Net with Attention

This model adds a spatial attention gate to each encoder-decoder skip connection. Each gate receives features from the encoder and a gating signal from the decoder, computes an attention map, and filters the skip features before concatenation.

The attention mechanism is intended to:

- emphasize relevant cell regions;
- suppress irrelevant background features;
- reduce false-positive detections caused by background noise;
- preserve smoother and more accurate boundaries in crowded scenes.

The ConvLSTM component remains unchanged so that temporal modeling is consistent with the baseline.

### 3. LSTM-ResU-Net

This model replaces the standard convolutional blocks in the encoder and decoder with residual blocks. A residual shortcut adds the block input to its processed output, improving information and gradient flow through the network.

The residual design is intended to:

- preserve small structures and thin boundaries;
- reduce excessive smoothing;
- improve learning in deeper feature-extraction stages;
- improve the detection of faint cells;
- support better separation of nearby cells.

The ConvLSTM component remains unchanged to preserve the temporal behavior of the original architecture.

## Dataset

The project uses the **Fluo-N2DH-GOWT1** dataset from the [Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/).

The dataset contains time-lapse fluorescence microscopy sequences of GFP-GOWT1 mouse stem cells. Each frame has a corresponding segmentation mask.

Only this cell type was used in the project as a proof of concept.

### Dataset Split

The frames were divided into three temporally continuous, non-overlapping subsets:

| Subset | Number of frames |
|---|---:|
| Training | 92 |
| Validation | 72 |
| Test | 20 |

Temporal continuity was preserved within each subset. Separate time ranges were used for training, validation, and testing to reduce temporal leakage between the sets.

## Preprocessing

Each microscopy frame was normalized independently using z-score normalization:

```text
normalized_image = (image - mean) / standard_deviation
```

This produces a more consistent intensity distribution across frames and improves training stability.

Training samples were created from sequences of consecutive frames and cropped into fixed-size image patches.

### Data Augmentation

Augmentations were applied only to the training data.

Spatial augmentations include:

- random cropping;
- horizontal and vertical flipping;
- rotations in multiples of 90 degrees;
- elastic deformation;
- affine transformation;
- random brightness adjustment;
- random contrast adjustment.

Temporal augmentations include:

- random reversal of frame order;
- temporal subsampling of the sequence.

## Model Input and Output

Each training sample contains a sequence of **four consecutive frames**. Every frame is cropped to:

```text
128 × 128 pixels
```

Depending on the selected configuration, tensors may use one of the following layouts:

```text
NCHW: [batch, time, channels, height, width]
NHWC: [batch, time, height, width, channels]
```

The network predicts three segmentation classes:

```text
0 — background
1 — cell interior
2 — cell contour / boundary
```

## Training Configuration

All three models were trained using the same dataset split, preprocessing pipeline, augmentation settings, and general training conditions to support a fair comparison.

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-5` |
| Batch size | 5 |
| Sequence length | 4 frames |
| Crop size | `128 × 128` |
| Training iterations | 1,000 |
| Validation interval | Every 100 iterations |
| Output classes | Background, cell interior, cell contour |

The original reference work used a substantially longer training schedule. The shorter 1,000-iteration setup used here was selected for a proof-of-concept comparison under limited computational resources.

## Loss Function

Training uses a **weighted categorical cross-entropy loss** for the three segmentation classes.

Class weighting helps address the imbalance between the large background area and the smaller cell-interior and cell-boundary regions. It also allows additional importance to be assigned to cell contours, which are essential for separating adjacent instances.

Invalid or unlabeled pixels can be excluded from the loss calculation.

## Inference and Post-processing

During inference, the model produces a softmax probability map for each class. The prediction pipeline includes:

1. loading a trained checkpoint;
2. preprocessing the input image sequence;
3. predicting class probabilities;
4. thresholding the cell-interior probability map;
5. refining cell edges;
6. filling holes in detected regions;
7. applying connected-component analysis;
8. filtering instances according to object size;
9. generating binary and instance-labeled masks.

The code also includes optional post-processing operations such as watershed-based separation, small-object removal, hole filling, and color-coded visualization of detected instances.

## Evaluation Metric

The models were evaluated using the **SEG score**, an instance-level segmentation metric used in cell-tracking evaluation.

The metric considers whether each ground-truth cell is correctly detected and measures the overlap between the predicted and ground-truth instance masks. A score closer to 1 indicates better segmentation performance.

## Results

The following mean SEG scores were obtained on the 20-frame test set:

### Example Segmentation Results

![Instance segmentation comparison across the ground-truth mask, baseline U-Net configuration, attention-based configuration, and residual configuration](images/segmentation_comparison.png)

*Example instance-segmentation outputs on a test frame. Each color represents a separate predicted cell instance.*

| Model | Mean SEG score |
|---|---:|
| LSTM-U-Net baseline | 0.5062 |
| LSTM-U-Net with Attention | 0.5171 |
| LSTM-ResU-Net | **0.5607** |

### Qualitative Observations

**LSTM-U-Net baseline**

- detected a moderate number of cells;
- provided a balance between detection rate and boundary quality;
- did not always reproduce smooth or exact cell contours.

**LSTM-U-Net with Attention**

- slightly improved the SEG score compared with the baseline;
- produced smoother masks and more accurate boundaries;
- missed more weak or low-contrast cells;
- detected fewer instances than the other models.

**LSTM-ResU-Net**

- achieved the highest SEG score;
- detected the largest number of cells, including faint instances;
- produced some masks with distorted shapes;
- still required refinement of cell-boundary accuracy.

A limitation shared by all three models was the tendency to merge two closely positioned cells into a single predicted instance.

## Discussion

Under the selected proof-of-concept conditions, LSTM-ResU-Net provided the strongest quantitative performance. Its residual blocks improved the detection of faint cells, which contributed to the highest SEG score.

The attention model produced smoother and more precise boundaries but sometimes suppressed weak cells together with irrelevant background features. This reduced its detection rate even though its SEG score was slightly better than the baseline.

The extended architectures contain more parameters than the baseline and may require:

- more training iterations;
- larger and more diverse datasets;
- learning-rate optimization;
- class-weight tuning;
- regularization;
- improved inference thresholds;
- more advanced instance-separation post-processing.

Because the dataset is relatively small and homogeneous, the reported results should be interpreted as a proof of concept rather than as fully optimized model performance.

## Future Work

Potential improvements include:

- training for substantially more iterations;
- performing systematic hyperparameter optimization;
- evaluating additional Cell Tracking Challenge datasets;
- improving the separation of touching cells;
- tuning class weights and decision thresholds;
- refining predicted boundaries;
- comparing computational cost and inference time;
- combining residual blocks and attention gates in a single **LSTM-ResU-Net with Attention** architecture.

A combined architecture may potentially use the stronger faint-cell detection of LSTM-ResU-Net together with the smoother boundary localization of the attention model.

## Requirements

The project was developed in Python 3 with TensorFlow and Keras in Google Colab.

Main dependencies include:

```text
TensorFlow
Keras
NumPy
OpenCV
SciPy
Pillow
scikit-image
Matplotlib
Pandas
Tifffile
TQDM
Imagecodecs
Requests
```

A local virtual environment can be created with:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the main packages:

```bash
pip install tensorflow numpy opencv-python scipy pillow scikit-image matplotlib pandas tifffile tqdm imagecodecs requests
```

TensorFlow, CUDA, and cuDNN versions must be mutually compatible when GPU acceleration is used.

## Important: Google Colab Export Structure

The uploaded `.py` files were exported from Google Colab notebooks. Each file contains notebook sections that were originally used to generate separate project modules, including:

```text
Networks.py
Network_Res.py
DataHandeling.py
Params.py
losses.py
utils.py
train2D.py
Inference2D.py
ColorMarks.py
```

Some sections remain commented because the original notebooks used Jupyter commands such as:

```python
%%writefile Networks.py
```

The exported files may also contain shell commands such as:

```python
!python train2D.py
!pip install imagecodecs
```

These commands work in Google Colab or Jupyter but are not valid inside a standard standalone Python script.

Therefore, the current files should primarily be treated as **Colab notebook exports and architecture records**. For conventional local execution, the relevant sections should be separated into their original Python modules and notebook shell commands should be executed from a terminal instead.

## Recommended Repository Structure

```text
project/
├── models/
│   ├── lstm_unet.py
│   ├── lstm_attention_unet.py
│   └── lstm_resunet.py
├── data_handling.py
├── losses.py
├── params.py
├── utils.py
├── train.py
├── inference.py
├── evaluate.py
├── requirements.txt
├── README.md
└── data/
    ├── images/
    └── masks/
```

## Reproducibility Notes

For a fair comparison, the following settings should remain identical across all models:

- training, validation, and test frame ranges;
- random seed;
- normalization method;
- augmentation settings;
- crop size;
- sequence length;
- batch size;
- optimizer and learning rate;
- number of iterations;
- class weights;
- inference thresholds;
- object-size filtering;
- post-processing operations;
- SEG metric implementation.

## Known Limitations

- Several dataset paths and hyperparameters are hard-coded in the exported scripts.
- The data-loading workflow expects a specific metadata and folder structure.
- Some SciPy namespace imports are deprecated and may require updating.
- Stateful ConvLSTM layers require correct state resetting between independent sequences.
- `NCHW` execution may require a compatible GPU; `NHWC` is generally safer on CPU.
- The experiment uses one relatively small and homogeneous cell dataset.
- The models were trained for only 1,000 iterations.
- Closely touching cells may be merged into one predicted mask.
- The reported architectures were not exhaustively optimized.

## Attribution and License

Parts of the implementation preserve notices and workflow conventions from the original code base. Retain all existing attribution notices when modifying or redistributing the code.

No explicit license was identified in the supplied files. Before publishing or redistributing the repository, verify the license of the original implementation and add an appropriate `LICENSE` file. Without an explicit license, reuse and redistribution rights are not automatically granted.
