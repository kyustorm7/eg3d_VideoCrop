# Code for Smooth Cropping for EG3D Inversion


### Setup & Requirements
NVIDIA GPUs. We have done all testings on RTX 3090 GPU.

64-bit Python 3.9, PyTorch 1.11.0 + CUDA toolkit 11.3

```
cd Deep3DFaceRecon_pytorch
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast    # ./Deep3DFaceRecon_pytorch/nvdiffrast
pip install .

cd ..    # ./Deep3DFaceRecon_pytorch
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/
```