# CSA-GAN

## Network Structure
![network_structure](./figures/frame.png)

## Main Requirements
* python 3.6+
* pytorch 1.0+
* numpy
* matplotlib
* opencv


## Prepare data
1. Download the preprocessed metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) and [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
2. Download [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset and extract the images to `data/birds/`
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`

## Pre-trained DAMSM model
1. Download the [pre-trained DAMSM](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V) for CUB and save it to `DAMSMencoders/`
2. Download the [pre-trained DAMSM](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ) for coco and save it to `DAMSMencoders/`

## Trained model
you can download our trained models from our [onedrive repo](https://www.aliyundrive.com/s/bd9SPBGtwZp)

## Start training
Run main.py file. Please adjust args in the file as your need.


## Evaluation
please run `IS.py` and `test_lpips.py` (remember to change the image path) to evaluate the `IS` and `diversity` scores, respectively.

For evaluating the `FID` score, please use this repo https://github.com/bioinf-jku/TTUR.



## Qualitative Results
Some qualitative results on coco and birds dataset from different methods are shown as follows:
![qualitative_results](./figures/res.jpg)





## Acknowledgements

This implementation borrows part of the code from SSA-GAN



