import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, 'data')
MALE_DIR = os.path.join(DATA_ROOT, 'male')
FEMALE_DIR = os.path.join(DATA_ROOT, 'female')
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
