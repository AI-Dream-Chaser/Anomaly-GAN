# Pytorch Anomaly-GAN


## Prerequisites
Code is intended to work with   
```Python 3.8.x```  
```CUDA Version 11.4```  
```torch Version 1.12.1```



### [PyTorch & torchvision](http://pytorch.org/)
Follow the instructions in [pytorch.org](http://pytorch.org) for your current setup



## Testing
### 1. Setup the dataset
First, you will need to download and setup a dataset. We provide the test data set link for download:
```
Link：https://pan.baidu.com/s/18Gyz-NKT6WRaoZtmAa_6zw   
Extract code： ed7f
```
Valid <dataset_name> are: screws_A, cloth_strips, etc. 

Alternatively you can build your own dataset by setting up the following directory structure:

    .
    ├── datasets                   
    |   ├── <dataset_name>         # i.e.
    |   |   ├── normal              # Testing
    |   |   |   ├── ImageSets       # Contains
    |   |   |   ├── JPEGImages      # Contains
    |   |   |   ├── mask               # Contains

### 3. Checkpoint!


The download link of the weight file (.pth) is as follows

```
Link：https://pan.baidu.com/s/1czemWm-v6zUOj97X-oVXzQ    
Extract code： cale
```
   
### 3. Test!
Test command line:
```
python ./test.py --dataroot datasets/<dataset_name>/ --category <category_name> --weight checkpoint/<weight_name>/ --output_path results/<path>/
```

## License
This project cooperates with Guangdong Pearl River Delta Intercity Railway Co., Ltd. - for details, please refer to the [License. md] (License. md) file

## Acknowledgments

The code is basically the implementation of the test file in [pythoch-Anomal-GAN]
. All credit goes to the authors of [Anomaly-GAN], Ruikang Liu, Weiming Liu, Zhongxin Zheng, Ling Wang, Liang Mao, Qisheng Qiu and Guangzheng Ling.
