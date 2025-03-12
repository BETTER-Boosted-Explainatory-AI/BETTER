# BETTER

## To create venv
python -m venv .venv

## Activate the venv
venv/Scripts/activate

## Download requierments 
pip install --no-cache-dir -r requirements.txt

## To run the server locally
uvicorn app:app --reload