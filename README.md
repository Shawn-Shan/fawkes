Fawkes
------

Fawkes is a privacy protection system developed by researchers at [SANDLab](http://sandlab.cs.uchicago.edu/), University of Chicago. For more information about the project, please refer to our project [webpage](http://sandlab.cs.uchicago.edu/fawkes/).  

We published an academic paper to summary our work "[Fawkes: Protecting Personal Privacy against Unauthorized Deep Learning Models](https://www.shawnshan.com/files/publication/fawkes.pdf)" at *USENIX Security 2020*. 

If you would like to use Fawkes to protect your images, please check out our binary implementation on the [website](http://sandlab.cs.uchicago.edu/fawkes/#code). 


Copyright
---------
This code is only for personal privacy protection or academic research. 

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
* `--separate_target`   : whether select separate targets for each faces in the diectory. 

### Example

`fawkes -d ./imgs --mode mid`

### Tips

- Select the best mode for your need. `Low` protection is effective against most model trained by individual trackers with commodity face recongition model. `mid` is robust against most commercial models, such as Facebook tagging system. `high` is robust against powerful modeled trained using different face recongition API. 
- The perturbation generation takes ~60 seconds per image on a CPU machine, and it would be much faster on a GPU machine. Use `batch-size=1` on CPU and `batch-size>1` on GPUs. 
- Turn on separate target if the images in the directory belong to different person, otherwise, turn it off. 

### How do I know my images are secure? 

We offer two ways to test the robustness of our detection and both of which requires certain level of coding experience. More details please checkout in [evaluation](https://github.com/Shawn-Shan/fawkes/tree/master/evaluation) directory. 


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
