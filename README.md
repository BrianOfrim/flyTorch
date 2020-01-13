# flyTorch

Modify the pycocotools site-package's cocoeval.py file's functions setDetParams and setKpParams by changing:
np.round(...) to np.round(...).astype(int) to ensure that the rounded value is an integer.
For pycocotools==2.0.0 this change should be applied to lines:  
507, 508, 518, 519
