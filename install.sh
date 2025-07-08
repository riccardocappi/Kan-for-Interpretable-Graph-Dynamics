PYTORCH_VERSION=2.3.1
TORCH_GEOMETRIC_VERSION=2.3.1

echo "Installing dependencies"

pip install --no-cache-dir torch==${PYTORCH_VERSION} --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==${TORCH_GEOMETRIC_VERSION}
pip install PyYAML==6.0.1
pip install "numpy<2"
pip install matplotlib
pip install numgraph
pip install optuna
pip install torchdiffeq
pip install dill
pip install pysr
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install git+https://github.com/TorchSpatiotemporal/tsl.git
pip install sympytorch
pip install scikit-image
