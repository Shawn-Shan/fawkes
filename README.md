Fawkes
------

Fawkes is a privacy protection system developed by researchers at [SANDLab](http://sandlab.cs.uchicago.edu/), University of Chicago. For more information about the project, please refer to our project [webpage](http://sandlab.cs.uchicago.edu/fawkes/). Contact as at fawkes-team@googlegroups.com. 

We published an academic paper to summarize our work "[Fawkes: Protecting Personal Privacy against Unauthorized Deep Learning Models](https://www.shawnshan.com/files/publication/fawkes.pdf)" at *USENIX Security 2020*. 

If you would like to use Fawkes to protect your identity, please check out our binary implementation on the [website](http://sandlab.cs.uchicago.edu/fawkes/#code). 


Copyright
---------
This code is intended only for personal privacy protection or academic research. 

We are currently exploring the filing of a provisional patent on the Fawkes algorithm. 

Usage
-----

`$ fawkes`

Options:

* `-m`, `--mode`       : the tradeoff between privacy and perturbation size
* `-d`, `--directory`  : the directory with images to run protection 
* `-g`, `--gpu`        : the GPU id when using GPU for optimization
* `--batch-size`       : number of images to run optimization together 
* `--format`      : format of the output image. 

when --mode is `custom`: 
* `--th`       : perturbation threshold
* `--max-step`       : number of optimization steps to run 
* `--lr`       : learning rate for the optimization
* `--feature-extractor` : name of the feature extractor to use
* `--separate_target`   : whether select separate targets for each faces in the directory. 

### Example

`fawkes -d ./imgs --mode mid`

### Tips
- The perturbation generation takes ~60 seconds per image on a CPU machine, and it would be much faster on a GPU machine. Use `batch-size=1` on CPU and `batch-size>1` on GPUs. 
- Turn on separate target if the images in the directory belong to different person, otherwise, turn it off. 
- Run on GPU. The current fawkes package and binary does not support GPU. To use GPU, you need to clone this, install the required packages in `setup.py`, and replace tensorflow with tensorflow-gpu. Then you can run fawkes by `python3 fawkes/protection.py [args]`. 

### How do I know my images are secure? 
We are actively working on this. Python script that can test the protection effectiveness will be ready shortly. 

Quick Installation
------------------

Install from [PyPI][pypi_fawkes]:

```
pip install fawkes
```

If you don't have root privilege, please try to install on user namespace: `pip install --user fawkes`.



### Citation
```
@inproceedings{shan2020fawkes,
  title={Fawkes: Protecting Personal Privacy against Unauthorized Deep Learning Models},
  author={Shan, Shawn and Wenger, Emily and Zhang, Jiayun and Li, Huiying and Zheng, Haitao and Zhao, Ben Y},
  booktitle="Proc. of USENIX Security",
  year={2020}
}
```
