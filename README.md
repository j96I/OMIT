# OMIT

## Activate virtual environment
Optional, use Python 'venv' to get fine grained control of the python enviroment.

```bash
.\venv\Scripts\activate
```

## TRAIN the model
This script assumes you have already cloned the 'store' repo at the same level as the OMIT repo.

```bash
cd src
python model_training.py
```

## Running WEB application
Create a web server which listens for connections on port 5000.
This assumes you have already done the installation, below.

```bash
cd src
python app.py
```

Then browse to: http://localhost:5000/

## Installing Python and PyTorch
Install python (python3) 3.8 or above, and pip (pip3), and required packages using the steps below OS:

On Enterprise Linux:
* Use CentOS8/RHEL8 or higher, the python with EL7 is too old.
```bash
sudo yum install python3 python3-pip
```

On Windows:
* Download & install Anaconda (and gitbash, optionally)
* Open Anaconda terminal (or gitbash)


Install Required Packages:
* Use the terminal to install our dependencies:
```bash
pip3 install flask scikit-learn pandas jupyter matplotlib

# if you have a GPU for use with pytorch:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# OR
# if you dont have a GPU:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
