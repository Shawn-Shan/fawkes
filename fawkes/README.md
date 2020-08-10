# Fawkes Binary

This application is built for individuals to cloak their images before uploading to the Internet. For more information about the project, please refer to our project [webpage](http://sandlab.cs.uchicago.edu/fawkes/).  

If you are a developer or researcher planning to customize and modify on our existing code. Please refer to [fawkes](https://github.com/Shawn-Shan/fawkes/tree/master/). 

### How to Setup

#### MAC:

* Create a directory and move all the images you wish to protect into that directory. Note the path to that directory (e.g. ~/Desktop/images). 
* Open [terminal](https://support.apple.com/guide/terminal/open-or-quit-terminal-apd5265185d-f365-44cb-8b09-71a064a42125/mac) and change directory to fawkes (the unzipped folder). 
* (If your MacOS is Catalina) Run `sudo spctl --master-disable` to enable running apps from unidentified developer. We are working on a solution to bypass this step. 
* Run `./protection-v0.3 -d IMAGE_DIR_PATH` to generate cloak for images in `IMAGE_DIR_PATH`. 
* When the cloaked image is generated, it will output a `*_min_cloaked.png` image in `IMAGE_DIR_PATH`. The generation takes ~40 seconds per image depending on the hardware. 


#### PC:
* Create a directory and move all the images you wish to protect into that directory. Note the path to that directory (e.g. ~/Desktop/images). 
* Open terminal(powershell or cmd) and change directory to protection (the unzipped folder). 
* Run `protection-v0.3.exe -d IMAGE_DIR_PATH` to generate cloak for images in `IMAGE_DIR_PATH`. 
* When the cloaked image is generated, it will output a `*_min_cloaked.png` image in `IMAGE_DIR_PATH`. The generation takes ~40 seconds per image depending on the hardware. 

#### Linux:
* Create a directory and move all the images you wish to protect into that directory. Note the path to that directory (e.g. ~/Desktop/images). 
* Open terminal and change directory to protection (the unzipped folder). 
* Run `./protection-v0.3 -d IMAGE_DIR_PATH` to generate cloak for images in `IMAGE_DIR_PATH`. 
* When the cloaked image is generated, it will output a `*_min_cloaked.png` image in `IMAGE_DIR_PATH`. The generation takes ~40 seconds per image depending on the hardware. 


More details on the optional parameters check out the [github repo](https://github.com/Shawn-Shan/fawkes/tree/master/). 

