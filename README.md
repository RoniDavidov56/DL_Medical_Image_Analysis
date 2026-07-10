# ConvLSTM U-Net Architectures for Sequence-Based Cell Segmentation

This repository contains three deep-learning architectures for semantic and instance-oriented segmentation of time-lapse microscopy image sequences. All models combine an encoder-decoder segmentation network with **ConvLSTM** layers to capture spatial and temporal information across consecutive frames.

The repository compares three related architectures:

1. **LSTM-U-Net Baseline** — a standard U-Net-style encoder-decoder with skip connections and ConvLSTM layers.
2. **LSTM-ResU-Net** — replaces standard convolutional blocks with residual blocks to improve feature propagation and preserve fine structures.
3. **LSTM-U-Net with Attention** — adds spatial attention gates to encoder-decoder skip connections so that the decoder can emphasize relevant foreground regions and suppress background noise.

## Repository Files

```text
.
├── lstm_u_net_baseline.py
├── lstm_resunet.py
├── lstm_u_net_+_attention.py
└── README.md
```

| File | Architecture | Main modification |
|---|---|---|
| `lstm_u_net_baseline.py` | LSTM-U-Net | Reference encoder-decoder model with ConvLSTM and standard skip connections |
| `lstm_resunet.py` | LSTM-ResU-Net | Residual convolutional blocks for improved gradient flow and detail preservation |
| `lstm_u_net_+_attention.py` | Attention LSTM-U-Net | Attention gates applied to skip features before decoder fusion |

## Model Overview

### 1. LSTM-U-Net Baseline

The baseline model extends a 2D U-Net with ConvLSTM layers in the encoder. It processes image sequences rather than isolated frames, allowing the network to use temporal context while producing pixel-wise segmentation outputs.

Main components:

- multi-level encoder-decoder architecture;
- ConvLSTM-based temporal feature extraction;
- encoder-decoder skip connections;
- batch normalization and Leaky ReLU activations;
- three-class softmax output for background, cell interior, and cell boundary;
- weighted cross-entropy loss for class imbalance.

### 2. LSTM-ResU-Net

The residual version replaces standard convolutional processing blocks with residual blocks. Identity shortcuts help information and gradients move through the network, which may reduce over-smoothing and improve the preservation of small cells and thin boundaries.

Main expected advantages:

- improved gradient propagation;
- stronger preservation of fine spatial details;
- reduced degradation in deeper feature extraction;
- improved separation of adjacent or touching cells.

### 3. LSTM-U-Net with Attention

The attention model introduces spatial attention gates at the encoder-decoder skip connections. Each gate combines an encoder feature map with a decoder gating signal and generates an attention mask before feature fusion.

Main expected advantages:

- suppression of irrelevant background features;
- increased focus on likely cell regions;
- sharper segmentation boundaries;
- fewer background false positives in crowded scenes.

## Processing Pipeline

The code includes the main stages required for sequence-based cell segmentation:

1. loading image sequences and segmentation annotations;
2. image normalization and sequence batching;
3. optional data augmentation, including flips, rotations, brightness and contrast changes, affine transformations, and elastic deformation;
4. training with weighted pixel-wise cross-entropy;
5. model checkpoint saving and restoration;
6. softmax inference for background, cell-interior, and cell-edge classes;
7. thresholding and morphological post-processing;
8. connected-component analysis and instance labeling;
9. evaluation using an instance-level SEG/IoU-based measure;
10. generation of color-coded ground-truth and prediction visualizations.

## Data Format

The implementation is designed for sequential microscopy images and corresponding segmentation masks. The data-loading code expects sequence metadata and image paths organized according to the reader configuration used in the scripts.

Typical input tensors follow one of these layouts:

```text
NCHW: [batch, time, channels, height, width]
NHWC: [batch, time, height, width, channels]
```

The default model configuration primarily uses `NCHW`. Verify the selected data format before training, especially when running on CPU or on platforms with limited `channels_first` support.

The segmentation output contains three channels:

```text
0 — background
1 — cell interior
2 — cell boundary/edge
```

## Requirements

The scripts use the following main packages:

```text
Python 3
TensorFlow 2.x
NumPy
OpenCV
SciPy
Pillow
scikit-image
pandas
tifffile
tqdm
matplotlib
imagecodecs
requests
```

A suitable environment can be created with:

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

Install the main dependencies:

```bash
pip install tensorflow numpy opencv-python scipy pillow scikit-image pandas tifffile tqdm matplotlib imagecodecs requests
```

> TensorFlow, CUDA, and cuDNN versions must be compatible when GPU acceleration is used.

## Important: Colab Export Structure

The three `.py` files were exported from Google Colab notebooks. Each file contains several notebook sections that were originally used to generate separate project modules such as:

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

Many of these sections are currently commented because the original notebooks used commands such as:

```python
%%writefile Networks.py
```

The exported files also contain notebook shell commands such as:

```python
!python train2D.py
!pip install imagecodecs
```

These commands work in Colab/Jupyter but are not valid in a standard Python script. Therefore, the uploaded files should be treated as **notebook exports and architecture records**, not as immediately executable standalone scripts.

For a conventional Git repository, extract the relevant commented sections into separate `.py` modules and replace notebook commands with terminal commands.

## Recommended Project Structure

```text
project/
├── models/
│   ├── lstm_unet.py
│   ├── lstm_resunet.py
│   └── lstm_attention_unet.py
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

## Training

After separating the notebook-generated modules and configuring the dataset paths and training parameters, training can be started from the repository root with:

```bash
python train2D.py
```

Training includes:

- sequence-based mini-batches;
- weighted cross-entropy loss;
- optional augmentation;
- validation during training;
- checkpoint and model-parameter saving;
- state handling for stateful ConvLSTM layers.

Dataset locations, crop size, sequence length, batch size, data format, learning rate, class weights, and output paths must be configured in the relevant parameter section before execution.

## Inference

After training, configure the model checkpoint and sequence paths, then run:

```bash
python Inference2D.py \
  --model_path /path/to/model \
  --sequence_path /path/to/images \
  --filename_format "t*.tif" \
  --data_format NCHW
```

The inference process:

- restores the trained model;
- normalizes and resizes input frames;
- predicts class-probability maps;
- thresholds the cell-interior class;
- refines edges and fills holes;
- filters connected components by object size;
- saves binary or instance-labeled masks.

The threshold values, expected input resolution, minimum and maximum cell sizes, and output paths should be adapted to the target dataset.

## Evaluation

The included evaluation code converts binary masks into labeled instances and calculates an instance-level SEG score based on overlap between ground-truth and predicted objects.

It also supports:

- removal of small objects;
- filling of small holes;
- watershed-based separation of touching cells;
- instance-count comparison;
- color-matched visualization of ground truth and predictions;
- reporting mean and standard deviation across evaluated images.

## Notes and Limitations

- Several paths and hyperparameters are hard-coded and must be changed before use.
- The scripts assume a specific metadata and filename organization inherited from the original cell-segmentation workflow.
- Some SciPy imports use deprecated namespace paths and may need updating for recent SciPy versions.
- Stateful ConvLSTM layers require careful state resetting between independent sequences.
- `NCHW` execution may require a compatible GPU configuration; `NHWC` is often safer on CPU.
- The attention and residual models should be compared using the same dataset split, preprocessing, augmentation, loss weights, and post-processing thresholds.
- Architectural improvements should be validated experimentally rather than assumed from the model design alone.

## Reproducible Comparison

For a fair comparison among the three architectures, keep the following settings identical:

- training, validation, and test sequences;
- random seed;
- crop size and sequence length;
- batch size and number of iterations;
- optimizer and learning rate;
- class weights;
- augmentation settings;
- inference threshold;
- object-size filtering;
- evaluation metric implementation.

Useful comparison measures include:

- validation loss;
- pixel-level accuracy or IoU;
- instance-level SEG score;
- number of false-positive and missed instances;
- ability to separate touching cells;
- inference time and memory consumption.

## Acknowledgment

Parts of the implementation retain author information and workflow conventions from the original code base. Preserve the original notices and provide appropriate attribution when redistributing or publishing derived code.

## License

No explicit license was identified in the supplied files. Before publishing the repository, verify the licensing terms of the original implementation and add an appropriate `LICENSE` file. Without a license, others do not automatically receive permission to copy, modify, or redistribute the code.
