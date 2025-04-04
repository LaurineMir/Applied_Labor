python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Create Jupyter kernel for venv
python -m ipykernel install --user --name .venv