# Keyfilter-Aware Real-Time UAV Object Tracking

Matlab implementation of Keyfilter-Aware UAV Object Tracker (KAOT).

## Description and Instructions

### How to run

- download matconvnet
- install.m - Compile libraries and download network
- run_KAOT.m

### Features

1. Deep CNN features. It uses matconvnet [1], which is included in external_libs/matconvnet/. The `imagenet-vgg-m-2048`network available at <http://www.vlfeat.org/matconvnet/pretrained/> was used. You can try other networks, by placing them in the feature_extraction/networks/ folder.
2. HOG features.
3. Colorspace features. Currently grayscale and RGB are implemented.

## Publications

@inproceedings{Li2020ICRA,
  title={Keyfilter-aware real-time uav object tracking},
  author={Li, Yiming and Fu, Changhong and Huang, Ziyuan and Zhang, Yinqiang and Pan, Jia},
  booktitle={IEEE International Conference on Robotics and Automation},
  year={2020}
}

@article{Li2020TMM,
  title={Intermittent Contextual Learning for Keyfilter-Aware UAV Object Tracking Using Deep Convolutional Feature},
  author={Li, Yiming and Fu, Changhong and Huang, Ziyuan and Zhang, Yinqiang and Pan, Jia},
  journal={IEEE Transactions on Multimedia},
  year={2020}
}

## Acknowledgements

This work borrowed the feature extraction modules from the STRCF tracker (<https://github.com/lifeng9472/STRCF>) and the parameter settings from BACF ([www.hamedkiani.com/bacf.html](http://www.hamedkiani.com/bacf.html)).