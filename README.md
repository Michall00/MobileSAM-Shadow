## TL;DR

This repository is a research fork of MobileSAM focused on fine-tuning
segmentation models to explicitly handle visual artifacts such as
shadows and reflections. The resulting models improve object removal
quality in mobile inpainting pipelines by enabling artifact-aware
segmentation.

## Origin of the Repository

This repository is a research fork of the original MobileSAM project:
https://github.com/ChaoningZhang/MobileSAM

The original architecture and inference pipeline remain unchanged.
All contributions introduced here are limited to dataset preparation,
training, and model export.

## Motivation

In object removal pipelines, segmentation models are typically trained
to isolate foreground objects only. However, in real-world images,
objects are often accompanied by visual artifacts such as shadows
and reflections.

If these artifacts are not explicitly segmented, they remain in the
image after inpainting, leading to visually implausible results.

This repository explores artifact-aware fine-tuning of MobileSAM
to improve downstream object removal quality in mobile applications.

## Datasets and Data Preparation

Fine-tuning of the MobileSAM model in this repository was performed using
datasets containing explicit annotations of visual artifacts such as
shadows and reflections, in addition to foreground object masks.

The following datasets were used:

### Used Datasets

- **SOBA**  
  Dataset containing objects and its related shadows  can be found in this repo:
  https://github.com/stevewongv/InstanceShadowDetection

- **DEROBA**  
  Dataset providing object with its reflection.  
  https://github.com/bcmi/Object-Reflection-Generation-Dataset-DEROBA


## Training and Fine-Tuning

Fine-tuning was performed on artifact-augmented datasets using the
original MobileSAM training pipeline, with modified supervision
targets that include artifact regions.

No architectural changes were introduced.

## Qualitative Results

<table>
<tr>
  <td>
    <em>Baseline segmentation (artifact remains)
  </td>
  <td>
  <em>Artifact-aware segmentation</em>
  </td>
</tr>
  <tr>
    <td align="center">
      <img src="docs/artifact_example_auto.png" width="720"/><br/>
    </td>
    <td align="center">
      <img src="docs/artifact_fixed_auto.png" width="720"/><br/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/artifact_example_dog.png" width="720"/><br/>
    </td>
    <td align="center">
      <img src="docs/artifact_fixed_dog.png" width="720"/><br/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/artifact_example_ballon.png" width="720"/><br/>
    </td>
    <td align="center">
      <img src="docs/artifact_fixed_ballon.png" width="720"/><br/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/artifact_example_boat.png" width="720"/><br/>
    </td>
    <td align="center">
      <img src="docs/artifact_fixed_boat.png" width="720"/><br/>
    </td>
  </tr>
</table>


## Installation and Dependencies

This repository uses **uv** for dependency management.

To install all required dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

## ONNX Export

This repository supports exporting **both components of MobileSAM**
to the ONNX format:

- image encoder
- prompt encoder + mask decoder

This enables fully on-device inference pipelines, where image embeddings
can be computed once and reused for multiple prompts.

### Exporting the Prompt Encoder and Mask Decoder

The prompt encoder and mask decoder can be exported using the original
MobileSAM export script:

```bash
python scripts/export_onnx_model.py \
  --checkpoint weights/mobile_sam.pt \
  --model-type vit_t \
  --output weights/mobile_sam_decoder.onnx
```

To export the image encoder:

```bash
python scripts/export_image_encoder_onnx.py \
  --checkpoint weights/mobile_sam.pt \
  --model-type vit_t \
  --output weights/mobile_sam_image_encoder.onnx
```
