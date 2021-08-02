# Folders and Files
## configs
This folder is for configs of different neural network structures and datasets (No need to change anything here)
## data
This folder contains the used datasets
## demo
Contains the codes for the original demo, and the new inference I created.
## docker
Not used
## docs
Not used
## local_configs
Folder for the configurations of used model and datasets
It is necessary to chagne configs in this file to use new model or new dataset
- #### \_base_
  Here there are the configurations files for the used models and the used datasets (see [how_to_run.md](how_to_use.md) file for more detailed instructions and info)
- #### segformer
  Here there are the main configuration files for the neural network structures (**B0, B1, ..., B5**) (see [how_to_run.md](how_to_use.md) file for more detailed instructions and info)
## mmseg
This is the main folder for most of the code
- #### apis
  codes for the process of building the neural networks and thier functions in the cases of (train, inference, test)
- #### core
  the code for the builder function (no need to edit)
- #### datasets
  this folder contains the main types of datasets, (if new dataset is needed, it should be added here and added to the __init__.py file also)
- #### models
  here are the codes for the decoder_heads and loss functions 
## pretrained
Here are the pretrained incoders ```(mit_b0.pth, ..., mit_b5.pth)```

## requirments
file for requirments for this repository
## resources
not important
## results
tha file where the results of inference is saved (the new created folders will be named as the current time, to avoid duplication)
## tests
Not used
## tools
Folder where training and testing codes can be found