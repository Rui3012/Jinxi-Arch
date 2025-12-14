# Jinxi-Arch

# Create env (choose Python 3.11; stable for most data stacks)
conda create -n jinxi-env python=3.11 -y

# Activate
conda activate jinxi-env

python -m pip install --upgrade pip

pip install pandas numpy openpyxl scikit-learn matplotlib

# Script

python jinxi_converted.py --input data/Jinxi_Model_Input.xlsx --outdir outputs


