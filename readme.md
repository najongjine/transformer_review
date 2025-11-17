py -3.10 -m uv venv venv

.\venv\Scripts\Activate.ps1

윈도우에서 CUDA 버전 확인 방법
nvcc --version

GPU 드라이버가 지원하는 최대 CUDA 버전
nvidia-smi

uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
