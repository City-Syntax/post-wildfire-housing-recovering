# post-wildfire-housing-recovering

> A scalable building-level observatory of how communities rebuild after wildfire.

Code accompanying the study *Unequal post-wildfire housing recovery follows income and insurance gradients*.

**Collaborating institutions:** National University of Singapore (NUS), Massachusetts Institute of Technology (MIT), Southeast University, Texas Tech University, Chongqing University, and Tongji University.

Wildfire recovery is often measured through aggregate losses or reconstruction totals. These measures can obscure where housing remains absent, where rebuilding changes the built environment, and which communities regain functioning housing most readily. This project transforms longitudinal Street View imagery into building-level recovery trajectories and connects visible reconstruction and vegetation change with neighborhood socioeconomic, environmental, credit, assistance, and insurance conditions.

The workflow is designed to reveal where recovery stalls, how rebuilding pathways diverge across communities, and which institutional conditions are associated with unequal recovery. By making these differences measurable at building and neighborhood scales, the project supports more targeted, risk-appropriate, and equitable post-disaster recovery planning.

## Longitudinal recovery examples

The longitudinal image sequences below illustrate the visible recovery outcomes represented in the workflow.

<table>
  <tr>
    <td width="50%" valign="top"><strong>Empty lot</strong><br><img src="assets/trajectory_empty_lot.png" width="100%" alt="Examples of buildings that remained empty lots after wildfire"></td>
    <td width="50%" valign="top"><strong>Rebuilt equal</strong><br><img src="assets/trajectory_rebuilt_equal.png" width="100%" alt="Examples of buildings rebuilt to a similar visible form after wildfire"></td>
  </tr>
  <tr>
    <td width="50%" valign="top"><strong>Rebuilt improved</strong><br><img src="assets/trajectory_rebuilt_improved.png" width="100%" alt="Examples of buildings visibly upgraded during rebuilding"></td>
    <td width="50%" valign="top"><strong>Rebuilt degraded</strong><br><img src="assets/trajectory_rebuilt_degraded.png" width="100%" alt="Examples of buildings rebuilt in a visibly reduced form"></td>
  </tr>
</table>

## Workflow

<p align="center">
  <img src="assets/workflow.svg" width="100%" alt="Workflow for observing and analyzing post-wildfire housing recovery">
</p>

The repository contains four focused analysis modules:

1. retrieve an authorized Street View image from a panorama reference;
2. classify the visible recovery trajectory from chronological images;
3. measure streetscape vegetation with semantic segmentation; and
4. estimate the rebuilding, upgrading and insurance-policy models used in the analysis.

## Repository structure

```text
wildfire-housing-recovery/
├── assets/
├── code/
│   ├── statistical_models.py
│   ├── streetview_api_example.py
│   ├── trajectory_classification_example.py
│   └── vegetation_measurement.py
├── examples/
│   └── streetview_input_template.csv
├── environment.yml
└── README.md
```

## Installation

```bash
conda env create -f environment.yml
conda activate wildfire-housing-recovery
```

## Usage

### Retrieve a Street View image

Supply a panorama reference obtained through your own authorized query.

```bash
export GOOGLE_MAPS_API_KEY="YOUR_KEY"
python code/streetview_api_example.py \
  --panoid PANO_ID_FROM_YOUR_OWN_QUERY \
  --heading 180 \
  --output local_data/example.jpg
```

### Classify a recovery trajectory

Pass two to eight chronological images of the same building. The script returns the trajectory, a confidence value and a brief visual rationale.

```bash
export ANTHROPIC_API_KEY="YOUR_KEY"
python code/trajectory_classification_example.py \
  --images local_data/pre.jpg local_data/post.jpg \
  --model MODEL_ID_AVAILABLE_TO_YOU
```

### Measure vegetation change

The primary workflow measures vegetation in every valid image and summarizes each building by the median pre-fire and post-fire fractions.

```bash
python code/vegetation_measurement.py \
  --pre-images local_data/pre_1.jpg local_data/pre_2.jpg \
  --post-images local_data/post_1.jpg local_data/post_2.jpg
```

### Estimate the statistical models

Run the baseline rebuilding model, the upgrading model and the joint policy model on a prepared building-level table.

```bash
python code/statistical_models.py \
  --data local_data/analysis_data.parquet \
  --output outputs
```

The prepared table must contain a unique fire identifier (`fire_id`), state FIPS code, one of the five documented trajectory labels, the seven baseline covariates, and the policy fields named in `code/statistical_models.py`. Both CSV and Parquet inputs are supported.

On Windows PowerShell, set API keys with `$env:GOOGLE_MAPS_API_KEY="YOUR_KEY"` and `$env:ANTHROPIC_API_KEY="YOUR_KEY"`.

## Input data

The scripts operate on locally supplied imagery and prepared analytical tables. Street View imagery must be accessed through the Google Maps Platform and used in accordance with the applicable terms. API credentials should be stored as environment variables and should never be committed to the repository.

## Project team

[City Syntax](https://github.com/City-Syntax)
