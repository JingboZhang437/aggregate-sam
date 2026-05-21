# SAM-based Aggregate Particle Segmentation and Morphological Parameter Measurement Tool

Leveraging the powerful [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), this tool enables fully automatic aggregate segmentation, interactive correction, and precise morphological parameter measurement, directly delivering engineering-grade aggregate characteristic data.


## Installation

The code recommends python 3.9.

```bash
git clone https://github.com/JingboZhang437/aggregate-sam.git
cd aggregate-sam
pip install -r requirements.txt
```

## Getting Started

Prepare your aggregate image for analysis and update the configuration synchronously:

```bash
image_path = r'1.jpg'
```

This tool uses a square cardboard under the aggregates for physical size calibration.
Please place a standard square cardboard below the aggregates when capturing images.
Measure its actual side length and set it in real_length (unit: mm).

```bash
real_length = 800
```

The program will automatically convert pixel measurements to real-world millimeters.

Download a [model checkpoint](https://github.com/JingboZhang437/aggregate-sam/new/master?filename=README.md#1-model-checkpoints) and modify this part in the code synchronously:

```bash
model_type = "vit_b"
sam_checkpoint1 = r"sam_vit_b_01ec64.pth"
```

Run the main program:

```bash
python sam_calculate_new.py
```

## Operation Workflow
### Step 1: 4-Point Perspective Calibration
- An image window will appear. Click to select **4 calibration corner points** in sequence.
- After selection, use the **arrow keys** to fine-tune the point positions.
- Press **ESC** to confirm, and the calibrated image will be generated automatically. Close the window.

### Step 2: Automatic Aggregate Segmentation & Mask Editing
- The program automatically uses SAM to perform fully automatic segmentation of aggregate particles.
- Automatically merges overlapping aggregate masks.
- An interactive window will pop up: **click on incorrectly segmented aggregates to remove them**, press **ESC** to finish editing. Close the window.
- Automatically deduplicates highly overlapping aggregate masks.

### Step 3: Aggregate Analysis Results Output
Once the program finishes, all engineering results are generated in the `output/` directory:
- `res_excel.xlsx`: Parameter table containing aggregate **major axis, minor axis, area, and equivalent volume**.
- `res_mask.pkl`: Binary file of the aggregate segmentation masks.
- `res_img/`: Visualization images of the entire processing workflow.

## Model Checkpoints

Three model versions are available with different backbone sizes. Click the links below to download the checkpoint for the corresponding model type.

- vit_h: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
  
- vit_l: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)

- vit_b: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Notes
This project is developed based on the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything). Sincere gratitude to Facebook Research for open-sourcing the SAM model.
