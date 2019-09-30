# epi-shift
Depth Estimation in Light Fields using a Recurrent Neural Network Architecture

# Requirements
* python3
* CUDA 9.2
* cuDNN 7.6
* [HCI 4D Lightfield Dataset](http://hci-lightfield.iwr.uni-heidelberg.de) 
* (virtualenv)

# Installation
For the installation I recommend virtualenv.
```sh
git clone git@github.com:titus-leistner/epi-shift.git
cd epi-shift/
python3 -m venv .
source bin/activate
pip install -r requirements.txt
```

# Usage
To train the model, run:
```sh
python train.py --tset path/to/hci4d/dataset/additional --vset path/to/hci4d/dataset/training --bsz [batch size]
```
Run `python train.py --help` to see all options.

To run inference on a dataset, run:
```sh
python infer.py --dset path/to/hci4d/dataset/test --params prms.pt
```
The disparity maps are saved as png and pfm files to ./out
