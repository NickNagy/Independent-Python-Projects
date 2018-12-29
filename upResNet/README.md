#upResNet

**edgeDetector**: edge detection algorithms.

**image_util**: for generating image batches. *Modified from https://github.com/jakeret/tf_unet to better fit upResNet

**layers**: variable, cropping and summary functions for upResNet. *Taken from different modules from https://github.com/jakeret/tf_unet

**resolutionDatabase**: for generating x, y and weight images for upResNet

**train_test_script**: script for creating, restoring, training, testing upResNet models

**upResNet**: fully-conv CNN for predicting higher res images from lower res ones.

**upResNet_v2**: version of upResNet that allows for multiple resizing options from the same model. Has many bugs.
