Fawkes Evaluation
-----------------


We offer two ways to test the protection is effective, 1) train a local face recognition model using transfer learning, 2) use Microsoft Azure API. 
Note that we can't guarantee the protection is always successful due to new development in face recognition technique. 

Evaluation with Local Model
---------------------------
To evaluate using local model, you are highly recommended to have a GPU device and train the model on it. Otherwise, the evaluation will be extremely slow and might even damage the CPUs on some machine. 

To evaluate, run `python3 eval_local.py -d IMAGE_DIR`. Where `IMAGE_DIR` is the image directory send to Fawkes protection code, and it must contain both original images (testing) and cloaked image (training). 
All images in the directory must belong to the same person and have at least 10 images in them. Also, you cannot turn on `--seperate-target` during the protection. (We are working on reducing some of these limitations.)
The script will output the protection success rate at the end. 


Evaluation with Microsoft Azure
---------------------------
forthcoming...

