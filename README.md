# Waymo Object Detection

This repository contains files, scripts, notebooks which demonstrate 2D object detection on Waymo Open Dataset. Steps include downloading raw Waymo data (v1.2.0), transforming it to tfrecords, utilizing transfer learning for training, evaluating the model, and running inference on video clips and images.

## Directory Structure
```
waymo-object-detection
│   data_download_preprocess.ipynb      # download/transform Waymo data
│   create_waymo_tfrecord.py            # transforms Waymo data
│   obj_detection_installation.ipynb    # install TF Obj. Detection
│   train_eval.ipynb                    # run training and evaluation
│   inference.ipynb                     # load saved model and infer
│   README.md                           # this file
│   
└───exportmodel_waymo_v120_efficientdet_d4    # dir containing exported TF model
    │   pipeline.config                       # config values for training, eval, etc.
    └───saved_model                           # TF loadable model
    │   │    ...
    └───checkpoint                            # checkpoint from last training run
        │    ...
```

## Downloading Waymo Data

Instructions to obtain access to Waymo data can be found with the following link: https://waymo.com/open/download/. The raw data was downloaded directly from Google Cloud Storage at "gs://waymo_open_dataset_v_1_2_0_individual_files".

## Transforming Raw Data to TFRecord format

The "create_waymo_tfrecord.py" script loads raw data in chunks and outputs transformed data to sharded tfrecord files (determined by num_shards parameter). The output format is a standard format used by the Tensorflow Object Detection API and more details can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md). The raw data includes images from multiple cameras and collected at 10Hz. During transformation, every 5th image from the FRONT camera is kept and rest of the images are discarded. This reduces the effective capture rate to 2Hz. Overall, the size of the raw training and validation dataset is reduced from just under 1TB to about 15GB.

## Tensorflow Object Detection Installation

Official [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md) are used to install the Object Detection API. One slight modification included enabling memory growth in Tensorflow through the [set_memory_growth](https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth) method. This helped avoid running out of GPU memory while initializing/training.

## Training

The pre-trained model obtained from [Tensorflow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) is **"EfficientDet D4 1024x1024"**. This model provides a good balance between performance and accuracy. Training was conducted with 50,000 total steps and a batch size of 14 using **7 Nvidia Tesla V100 32GB GPUs**. The full training (and evaluation) parameters/values can be found in the [pipeline.config](exportmodel_waymo_v120_efficientdet_d4/pipeline.config) file.

## Evaluation

COCO evaluation metrics are used. The results are summarized in the following table:

| | | |
| --- | --- | --- |
| Average Precision (AP) |[ IoU=0.50:0.95 \| area= all \| maxDets=100 ]|0.225|
| Average Precision (AP) |[ IoU=0.50 \| area= all \| maxDets=100 ]|0.415
| Average Precision (AP) |[ IoU=0.75 \| area= all \| maxDets=100 ]|0.216
| Average Recall (AR) |[ IoU=0.50:0.95 \| area= all \| maxDets= 1 ]|0.088
| Average Recall (AR) |[ IoU=0.50:0.95 \| area= all \| maxDets= 10 ]|0.236
| Average Recall (AR) |[ IoU=0.50:0.95 \| area= all \| maxDets=100 ]|0.294

## Inference
Using cv2, inference can be done on a video by analyzing each frame and outputting to another video. Performance was about **6.94 frames per second** or about **144ms/frame**.
