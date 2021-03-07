Fawkes
------
:warning: Check out our MacOS/Windows Software on our official [webpage](https://sandlab.cs.uchicago.edu/fawkes/#code).

Fawkes is a privacy protection system developed by researchers at [SANDLab](https://sandlab.cs.uchicago.edu/),
University of Chicago. For more information about the project, please refer to our
project [webpage](https://sandlab.cs.uchicago.edu/fawkes/). Contact us at fawkes-team@googlegroups.com.

We published an academic paper to summarize our
work "[Fawkes: Protecting Personal Privacy against Unauthorized Deep Learning Models](https://www.shawnshan.com/files/publication/fawkes.pdf)"
at *USENIX Security 2020*.


Copyright
---------
This code is intended only for personal privacy protection or academic research.

Usage
-----

`$ fawkes`

Options:

* `-m`, `--mode`       : the tradeoff between privacy and perturbation size. Select from `low`, `mid`, `high`. The
  higher the mode is, the more perturbation will add to the image and provide stronger protection.
* `-d`, `--directory`  : the directory with images to run protection.
* `-g`, `--gpu`        : the GPU id when using GPU for optimization.
* `--batch-size`       : number of images to run optimization together. Change to >1 only if you have extremely powerful
  compute power.
* `--format`      : format of the output image (png or jpg).

### Example

`fawkes -d ./imgs --mode low`
or 
`python3 protection.py -d ./imgs --mode min`


### Tips

- The perturbation generation takes ~60 seconds per image on a CPU machine, and it would be much faster on a GPU
  machine. Use `batch-size=1` on CPU and `batch-size>1` on GPUs.
- Run on GPU. The current Fawkes package and binary does not support GPU. To use GPU, you need to clone this repo, install
  the required packages in `setup.py`, and replace tensorflow with tensorflow-gpu. Then you can run Fawkes
  by `python3 fawkes/protection.py [args]`.

![](http://sandlab.cs.uchicago.edu/fawkes/files/obama.png)

### How do I know my images are secure?

We are actively working on this. Python scripts that can test the protection effectiveness will be ready shortly.

Quick Installation
------------------

Install from [PyPI](https://pypi.org/project/fawkes/):

```
pip install fawkes
```

If you don't have root privilege, please try to install on user namespace: `pip install --user fawkes`.


Academic Research Usage
-----------------------
For academic researchers, whether seeking to improve fawkes or to explore potential vunerability, please refer to the
following guide to test Fawkes.

To protect a class in a dataset, first move the label's image to a seperate location and run Fawkes. Please
use `--debug` option and set `batch-size` to a reasonable number (i.e 16, 32). If the images are already cropped and
aligned, then also use the `no-align` option.

### Citation

```
@inproceedings{shan2020fawkes,
  title={Fawkes: Protecting Personal Privacy against Unauthorized Deep Learning Models},
  author={Shan, Shawn and Wenger, Emily and Zhang, Jiayun and Li, Huiying and Zheng, Haitao and Zhao, Ben Y},
  booktitle={Proc. of {USENIX} Security},
  year={2020}
}
```
