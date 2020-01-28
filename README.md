# flyTorch

**This project is now deprecated, it's functionality has been added to https://github.com/BrianOfrim/boja**

Heavily inspired by:
Torchvision detection reference: https://github.com/pytorch/vision/tree/master/references/detection
Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

# Getting Started
Modify the pycocotools site-package's cocoeval.py file's functions setDetParams and setKpParams by changing:
np.round(...) to int(np.round(...)) to ensure that the rounded value is an integer.
For pycocotools==2.0.0 this change should be applied to lines:  
507, 508, 518, 519
