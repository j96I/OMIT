# OMIT

## Activate virtual environment
(optional)

```bash
.\venv\Scripts\activate
```

## TRAIN the model 
(this assumes you have cloned the 'store' repo at the same level as the OMIT repo.)

```bash
cd src
python train.py
```

Then browse to: http://localhost:5000/


## Running WEB application 
(assumes you have already done the installation and training)

```bash
cd src
python app.py
```

Then browse to: http://localhost:5000/

## Installation
Install python (python3) 3.8 or above, and pip (pip3).

On Enterprise Linux (CentOS8/RHEL8 or higher, the python with EL7 is too old):
```bash
sudo yum install python3 python3-pip
```

On Windows:
Download & install Anaconda (and gitbash, optionally)
Open Anaconda terminal (or gitbash)

Use the terminal to install our dependencies:

```bash
pip3 install flask scikit-learn pandas jupyter matplotlib

# if you have a GPU for use with pytorch:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# if you dont have a GPU:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
