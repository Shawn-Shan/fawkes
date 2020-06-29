# Fawkes
Code implementation of the paper "[Fawkes: Protecting Personal Privacy against Unauthorized Deep Learning Models](https://arxiv.org/pdf/2002.08327.pdf)", at *USENIX Security 2020*. 

### BEFORE YOU RUN OUR CODE
We appreciate your interest in our work and for trying out our code. We've noticed several cases where incorrect configuration leads to poor performances of protection. If you also observe low detection performance far away from what we presented in the paper, please feel free to open an issue in this repo or contact any of the authors directly. We are more than happy to help you debug your experiment and find out the correct configuration. 

### ABOUT

This repository contains code implementation of the paper "[Fawkes: Protecting Personal Privacy against Unauthorized Deep Learning Models](https://arxiv.org/pdf/2002.08327.pdf)", at *USENIX Security 2020*. 

### DEPENDENCIES

Our code is implemented and tested on Keras with TensorFlow backend. Following packages are used by our code.

- `keras==2.3.1`
- `numpy==1.18.4`
- `tensorflow-gpu==1.13.1`

Our code is tested on `Python 3.6.8`

### HOWTO

#### Download and Config Datasets
The first step is to download several datasets for protection and target selection. 
1. Download the following dataset to your local machine. After downloading the datasets, restructure it the same way as the FaceScrub dataset downloaded. 
    - FaceScrub -- used for protection evaluation (link)
    - VGGFace1 -- used for target select (link)
    - VGGFace2 -- used for target select (link)
    - WebFace -- used for target select (link)

2. Config datasets
open `fawkes/config.py` and update the `DATASETS` dictionary with the path to each dataset. Then run `python fawkes/config.py`. Every time the datasets are updated or moved, remember to rerun the command with the updated path. 

3. Calculate embeddings using feature extractor. 
Run `python3 fawkes/prepare_feature_extractor.py --candidate-datasets scrub vggface1 vggface2 webface`. This will calculate and cache the embeddings using the default feature extractor we provide. To use a customized feature extractor, please look at the Advance section at the end. 

#### Generate Cloak for Images
To generate cloak, run 
`python3 fawkes/protection.py --gpu 0 --dataset scrub --feature-extractor webface_dense_robust_extract`
For more information about the detailed parameters, please read `fawkes/protection.py`. 
The code will output a directory in `results/` with `cloak_data.p` inside. You can check the cloaked images or inspect the changes in `this notebook`. 

#### Evaluate Cloak Effectiveness
To evaluate the cloak, run `python3 fawkes/eval_cloak.py --gpu 0 --cloak_data PATH-TO-RESULT-DIRECTORY --transfer_model vggface2_inception_extract`. 

The code will print out the tracker model accuracy on uncloaked/original test images of the protected user, which should be close to 0. 


#### Exisiting Feature extractors

We shared three different feature extractors under feature_extractors/
1. low_extract.h5: trained on WebFace dataset with DenseNet architecture. 
2. mid_extract.h5: VGGFace2 dataset with DenseNet architecture. Trained with PGD adversarial training for 5 epochs. 
3. high_extract.h5: WebFace dataset with DenseNet architecture. Trained with PGD adversarial training for 20 epochs. 

### Citation
```
@inproceedings{shan2020fawkes,
  title={Fawkes: Protecting Personal Privacy against Unauthorized Deep Learning Models},
  author={Shan, Shawn and Wenger, Emily and Zhang, Jiayun and Li, Huiying and Zheng, Haitao and Zhao, Ben Y},
  booktitle="Proc. of USENIX Security",
  year={2020}
}
```