# Fawkes Binary

This application is built for individuals to cloak their images before uploading to the Internet. For more information about the project, please refer to our project [webpage](http://sandlab.cs.uchicago.edu/fawkes/).  

If you are a developer or researcher planning to customize and modify on our existing code. Please refer to [fawkes](https://github.com/Shawn-Shan/fawkes/tree/master/). 

### How to Setup

#### MAC:

* Download the binary following this [link](http://sandlab.cs.uchicago.edu/fawkes/files/fawkes_binary.zip) and unzip the download file. 
* Create a directory and move all the images you wish to protect into that directory. Note the path to that directory (e.g. ~/Desktop/images). 
* Open [terminal](https://support.apple.com/guide/terminal/open-or-quit-terminal-apd5265185d-f365-44cb-8b09-71a064a42125/mac) and change directory to fawkes (the unzipped folder). 
* (If your MacOS is Catalina) Run `sudo spctl --master-disable` to enable running apps from unidentified developer. 
* Run `./fawkes -d IMAGE_DIR_PATH -m low` to generate cloak for images in `IMAGE_DIR_PATH`. 
* More details on the optional parameters check out the [github repo](https://github.com/Shawn-Shan/fawkes/tree/master/). 


#### PC:
More details coming soon. The steps should be similar to the MAC setup. 

