PYTHON_VERSION=python3.9
PYTORCH_VERSION=2.3.1
TORCH_GEOMETRIC_VERSION=2.3.1

echo "Creating environment"
$PYTHON_VERSION -m venv .venv/gkanconv-ode
source .venv/gkanconv-ode/bin/activate

echo "Environment created and activated"

pip install --no-cache-dir torch==${PYTORCH_VERSION} --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==${TORCH_GEOMETRIC_VERSION}
pip install PyYAML==6.0.1
pip install "numpy<2"
pip install matplotlib
pip install numgraph
pip install optuna
pip install torchdiffeq