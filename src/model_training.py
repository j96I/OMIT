from utils.setup_validation import setup_validate
from omit import train_model

print("OMIT - Pre-training Environment Check:\n")
setup_validate()

print("\nOMIT - Training Model:\n")
# Assume we have already cloned the 'store' repo at the same 
# directory level as this repo was cloned.
train_model(retrain=False, img_dir='../store')

print("\nOMIT - Done.\n")
