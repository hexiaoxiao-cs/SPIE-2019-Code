# SPIE-2019-Code
This repository contains code for Effective 3D humerus and scapula extraction using low-contrast and high-shape-variability MR data

The data for training the network can not be published by the regulation of clinics and Patient Privacy laws.

Please contact:
  - \* Kang Li kang.li@rutgers.edu
  - Xiaoxiao He xh172@scarletmail.rutgers.edu
  - Chaowei Tan chaoweitan@gmail.com
  - "\*" for the corresponding author

## For using this code
  ### We encourage you to use the anaconda for python environment
  ### The following packages are used in the program:
      - SimpleITK
      - keras
      - nibabel
      - pytables
      - nilearn
      - nipype
      - tqdm
  ### Install [ANTs N4BiasFieldCorrection](https://github.com/stnava/ANTs/releases) and add the location of the ANTs 
binaries to the PATH environmental variable.
  ### Add the repository directory to the ```PYTONPATH``` system variable:
```
$ export PYTHONPATH=${PWD}:$PYTHONPATH
```
  ### Goto Preprocessing folder and run
  ```
  python preprocessing.py
  ```
  ### Goto Training folder and run
  ```
  python auto-augmentation.py
  ```
