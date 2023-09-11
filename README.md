# Yolov5 Nodes

Learning Loop Trainer and Detector Node for Yolov5 (object detection and classification of images). The DL part is based on https://github.com/ultralytics/yolov5
This repository is an implementation of Nodes that interact with the [Zauberzeug Learning Loop](https://github.com/zauberzeug/learning_loop_node).


# Trainer

This node is used to train Yolov5 Models in the Learning Loop. It is based on [this image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-07.html) running Python 3.10.

## Images

Trainer Docker-Images are published on https://hub.docker.com/r/zauberzeug/yolov5-trainer

Images can be pulled with `docker pull zauberzeug/yolov5-trainer:latest`.
The script `docker.sh` in the folder `training` is recommended used to interact with this image. 
It is required to setup a .env file in the training folder with the following values

- HOST=learning-loop.ai"
- ORGANIZATION=zauberzeug"
- PROJECT=demo"
- YOLOV5_MODE=CLASSIFICATION | DETECTION

# Detector

## Images

Detector Images are published on https://hub.docker.com/r/zauberzeug/yolov5-detector

Images can be pulled with `docker pull zauberzeug/yolov5-detector:32.6.1`, where `32.6.1` is the used `Tag`(see https://hub.docker.com/r/zauberzeug/yolov5-detector/tags). It corresponds to the L4T version. Right now, `32.6.1` and `32.5.0` are supported.

Pulled images can be run with the `docker.sh` script by calling `./docker.sh run-image`.
Local builds can be run with `./docker.sh run`
Mandatory parameters (Please adapt as needed):

- HOST=learning-loop.ai"
- ORGANIZATION=zauberzeug"
- PROJECT=demo"
- YOLOV5_MODE=CLASSIFICATION | DETECTION

## Publish a new release

```
# build docker image
./docker.sh b

# publish docker image
./docker.sh p
```

## Get Detections

### Curl

```
curl --request POST -H 'mac: FF:FF:FF:FF:FF' -F 'file=@test.jpg' http://localhost:8004/detect
```

### Python

```
headers = {'mac': '0:0:0:0', 'tags':  'some_tag'}
with open('test.jpg', 'rb') as f:
        data = [('file', f)]
        response = requests.post(
            'http://localhost:8004/detect', files=data, headers=headers)
```

# Formats

The trainer uses the `yolov5_pytorch` format identifier (`yolov5_cla_pytorch` for classification).
When it saves a model to the Learning Loop it saves the model as `yolov5_pytorch` and `yolov5_wts` (respectively `yolov5_cla_pytorch` and `yolov5_cla_wts` for classification).
The wts formats may be used by a detector wunning on a NVIDIA jetson device to create an engine file as required by tensorrtx (see https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5).
