#!/bin/bash
set -e
######################################################
# when copy from tpu_compile need add option as below:
# 1.add INPUT_SHAPE
# 2.modify INPUT_SCALE (each channel)
# 3.modify CALI_TABLE path (don't include input num)
# 4.add CALI_IMAGES
# 5.add IMAGE_PATH (default cat.jpg)
# 6.modify TOLERANCE(remove duplicate cos sim)
######################################################

#default values
export MODEL_TYPE="caffe"   # caffe, pytorch, onnx, tflite, tf
export STD=1,1,1
export MODEL_CHANNEL_ORDER="bgr"
export EXCEPTS=-
export EXCEPTS_BF16=-
export CALIBRATION_IMAGE_COUNT=1000
export DO_QUANT_INT8=1
export DO_FUSED_PREPROCESS=1
export DO_FUSED_POSTPROCESS=0
export DO_ACCURACY_FP32_INTERPRETER=0
export DO_ACCURACY_FUSED_PREPROCESS=0
export EVAL_MODEL_TYPE="imagenet"
export LABEL_FILE=$REGRESSION_PATH/data/synset_words.txt
export BGRAY=0
export RESIZE_KEEP_ASPECT_RATIO=0
export TOLERANCE_FP32=0.999,0.999,0.998
export TOLERANCE_INT8_CMDBUF=0.99,0.99,0.99
export DO_QUANT_BF16=1
export INT8_MODEL=0
export MIX_PRECISION_TABLE='-'1
export INPUT_NUM=100

# default inference and test image
export IMAGE_PATH=$REGRESSION_PATH/cv18xx_porting/data/cat.jpg

if [ $NET = "googlenet" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/googlenet/caffe/deploy_bs1.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/googlenet/caffe/bvlc_googlenet.caffemodel
# replace $REGRESSION_PATH/data/cali_tables/ with $REGRESSION_PATH/cv18xx_porting/cali_tables/
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]  # new attr
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=104,117,123
export INPUT_SCALE=1.0,1.0,1.0      # value per-channel
export INPUT=data
export OUTPUTS=prob
export TOLERANCE_INT8=0.975,0.768     # 2 value
export TOLERANCE_BF16=0.998,0.990     # 2 value
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export EXCEPTS=prob
export MODEL_CHANNEL_ORDER=bgr      # set channel order
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012 # set calibration dateset
fi

if [ $NET = "resnet50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/resnet/caffe/ResNet-50-model.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=104.01,116.67,122.68  # from ilsvrc_2012_mean.npy
export INPUT_SCALE=1.0,1.0,1.0
export MODEL_CHANNEL_ORDER="bgr"
export INPUT=input
export OUTPUTS=fc1000
export TOLERANCE_INT8=0.96,0.71
export TOLERANCE_MIX_PRECISION=0.96,0.95,0.73
export MIX_PRECISION_BF16_LAYER_NUM=10
export EXCEPTS=prob,res2c_relu,res3d_relu,res4f_relu
export TOLERANCE_BF16=0.99,0.96
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
fi

if [ $NET = "squeezenet_v1.0" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.0.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.0.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,227,227]]
export NET_INPUT_DIMS=227,227
export IMAGE_RESIZE_DIMS=227,227
export RAW_SCALE=255.0
export MEAN=104,117,123
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=data
export OUTPUTS=pool10
export TOLERANCE_INT8=0.973,0.762
export TOLERANCE_BF16=0.999,0.987
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.93
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "mobilenet_v1" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v1/caffe/mobilenet_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v1/caffe/mobilenet.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export INPUT=input
export OUTPUTS=fc7
export TOLERANCE_INT8=0.96,0.73
export TOLERANCE_BF16=0.99,0.92
export EXCEPTS=prob
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "mobilenet_v2" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export MODEL_CHANNEL_ORDER="bgr"
export INPUT=input
export OUTPUTS=fc7
export TOLERANCE_INT8=0.94,0.66
export TOLERANCE_BF16=0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export EXCEPTS=prob
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "inception_v3" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,299,299]]
export NET_INPUT_DIMS=299,299
export IMAGE_RESIZE_DIMS=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125,0.0078125,0.0078125
export INPUT=input
export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8=0.95,0.68
export TOLERANCE_BF16=0.99,0.93
export TOLERANCE_BF16_CMDBUF=0.99,0.94
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi

if [ $NET = "inception_v4" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/inception_v4/caffe/deploy_inception-v4.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/inception_v4/caffe/inception-v4.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,299,299]]
export NET_INPUT_DIMS=299,299
export IMAGE_RESIZE_DIMS=299,299
export RAW_SCALE=255.0
export MEAN=128.0,128.0,128.0
export INPUT_SCALE=0.0078125,0.0078125,0.0078125
export INPUT=input
export OUTPUTS=classifier
# export EXCEPTS=prob
export TOLERANCE_INT8=0.92,0.60
export TOLERANCE_BF16=0.99,0.89
export TOLERANCE_BF16_CMDBUF=0.99,0.93
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
export INPUT_NUM=100
fi

if [ $NET = "densenet_201" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_201.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/densenet/caffe/DenseNet_201.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export MODEL_CHANNEL_ORDER="bgr"
export INPUT=input
export OUTPUTS=fc6
export TOLERANCE_INT8=0.76,0.28
export TOLERANCE_BF16=0.99,0.92
export TOLERANCE_BF16_CMDBUF=0.99,0.96
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
export INPUT_NUM=100
fi

if [ $NET = "resnext50" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/resnext/caffe/resnext50-deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/resnext/caffe/resnext50.caffemodel
export LABEL_FILE=$MODEL_PATH/imagenet/resnext/caffe/corresp.txt
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MODEL_CHANNEL_ORDER="rgb"
export MEAN=0.0,0.0,0.0
export INPUT_SCALE=1.0,1.0,1.0
export INPUT=data
export OUTPUTS=prob
export EXCEPTS=data
export TOLERANCE_INT8=0.96,0.70
export TOLERANCE_BF16=0.99,0.95
export TOLERANCE_BF16_CMDBUF=0.99,0.97
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
export INPUT_NUM=1000
fi

if [ $NET = "vgg16" ]; then
export MODEL_DEF=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers_deploy.prototxt
export MODEL_DAT=$MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers.caffemodel
export CALI_TABLE=$REGRESSION_PATH/cv18xx_porting/cali_tables/${NET}_calibration_table
export INPUT_SHAPE=[[1,3,224,224]]
export NET_INPUT_DIMS=224,224
export IMAGE_RESIZE_DIMS=256,256
export RAW_SCALE=255.0
export MEAN=103.94,116.78,123.68
export INPUT_SCALE=0.017,0.017,0.017
export INPUT=input
export OUTPUTS=fc8
export TOLERANCE_INT8=0.997,0.928
export TOLERANCE_BF16=0.999,0.994
export TOLERANCE_BF16_CMDBUF=0.99,0.99,0.96
export MODEL_CHANNEL_ORDER=bgr
export EXCEPTS=prob
export CALI_IMAGES=$DATA_SET/imagenet/img_val_extracted/ILSVRC2012
fi
