# DRDmSR

DRDmSR is the official implementation of the paper Joint Demosaicking and Super Resolution for SFA Images.

Paper: [Joint Demosaicking and Super Resolution for SFA Images](https://ieeexplore.ieee.org/abstract/document/10838565)

Overview
- **What:** A PyTorch implementation that jointly performs demosaicking and super-resolution for single-frame active (SFA) images.
- **Who:** Research code and models for the DRDmSR project.

Highlights
- End-to-end model for demosaicking + super-resolution
- Training and inference scripts included

Quick start
1. Install dependencies:

	pip install -r requirements.txt

2. Run inference on real images:

	python inference_real_images.py --input <INPUT_PATH> --output <OUTPUT_PATH>

3. Train or fine-tune models:

	python train_demosaic.py
	python train_fine_tuning.py

Dataset
- This work uses the ARAD-1K dataset. You can download it here:

  https://drive.google.com/file/d/1rjTNNfw-h1z3Rf-UvwwvjD3ytOZZy3T2/view

Model weights
- Pre-trained model weights are private and are not included in this repository. If you need access for collaboration or evaluation, please open an issue or contact the repository owner.

Usage notes
- See the `test_data` and `test_demosaic_result` folders for example inputs/outputs.
- Scripts available: `inference_real_images.py`, `train_demosaic.py`, `train_fine_tuning.py`, and `train_PPI.py`.

Contributing
- Contributions are welcome. Please open an issue to discuss major changes before sending a pull request.

License
- See the `LICENSE` file for licensing information.

If you find this repository useful, please give it a star :D

