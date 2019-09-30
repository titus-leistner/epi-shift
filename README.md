# EPI-Shift
Learning to Think Outside the Box: Wide-Baseline Light Field Depth Estimation with EPI-Shift

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
The disparity maps are saved as png and pfm files to `./out`.

# Paper and Citation
The [paper](https://arxiv.org/pdf/1909.09059.pdf) was published at [3DV 2019](http://3dv19.gel.ulaval.ca) with an oral presentation. See the [project page](https://titus-leistner.de/learning-to-think-outside-the-box-wide-baseline-light-field-depth-estimation-with-epi-shift.html).

```bibtex
@inproceedings{Leistner2019,
  title = {Learning to Think Outside the Box: Wide-Baseline Light Field Depth Estimation with EPI-Shift},
  author = {Leistner, Titus and Schilling, Hendrik and Mackowiak, Radek and Gumhold, Stefan and Rother, Carsten},
  booktitle = {International Conference on 3D Vision (3DV)},
  doi = {10.1109/3DV.2019.00036},
  month = {sep},
  year = {2019},
}
```