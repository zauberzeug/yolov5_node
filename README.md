# Yolov5 Nodes

Learning Loop Trainer and Detector Node for Yolov5 (object detection and classification of images). The DL part is based on https://github.com/ultralytics/yolov5
This repository is an implementation of Nodes that interact with the Zauberzeug Learning Loop using the [Zauberzeug Learning Loop Node Library](https://github.com/zauberzeug/learning_loop_node).

# Trainer

This node is used to train Yolov5 Models in the Learning Loop. It is based on [this image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-07.html) running Python 3.10.

## Hyperparameters

We support all native hyperparameters of YOLOv5 (cf. `hyp_det.yaml` / `hyp_cla.yaml` for reference).
In addition, we support the following hyperparameters:

- `epochs`: The number of epochs to train the model.
- `detect_nms_conf_thres`: The confidence threshold for the NMS during inference and validation (not relevant for training).
- `detect_nms_iou_thres`: The IoU threshold for the NMS during inference and validation (not used for training).

Further, we support the following hyperparameters for point detection:

- `reset_points`: Whether to reset the size of the points after data augmentation.
- `point_sizes_by_id`: A dictionary that maps from point category uuids to the size of the points in the output (fractional size 0-1).
- `flip_label_pairs`: A list of pairs of point uuids that should be swapped when a horizontal flip is applied during data augmentation.

## Images

Trainer Docker-Images are published on https://hub.docker.com/r/zauberzeug/yolov5-trainer

New images can be pulled with `docker pull zauberzeug/yolov5-trainer:nlvX.Y.Z`, where `X.Y.Z` is the version of the node-lib used.

During development, i.e. when building the container from code it is recommended to use the script `docker.sh` in the folder `training` to build/start/interact with the image.
When using the script it is required to set up a .env file in the training folder that contains the loop-related configuration. The following variables should be set (note that some are inherited from the [Zauberzeug Learning Loop Node Library](https://github.com/zauberzeug/learning_loop_node) ):

| Name                   | Purpose                                              | Value                       | Default | Requi. only with ./docker.sh |
| ---------------------- | ---------------------------------------------------- | --------------------------- | ------- | ---------------------------- |
| YOLOV5_MODE            | Mode of the trainer                                  | CLASSIFICATION or DETECTION | -       | No                           |
| TRAINER_NAME           | Will be the name of the container                    | String                      | -       | Yes                          |
| LINKLL                 | Link the node library into the container?            | TRUE/FALSE                  | FALSE   | Yes                          |
| UVICORN_RELOAD         | Enable hot-reload                                    | TRUE/FALSE/0/1              | FALSE   | No                           |
| RESTART_AFTER_TRAINING | Auto-restart after training                          | TRUE/FALSE/0/1              | FALSE   | No                           |
| KEEP_OLD_TRAININGS     | Do not remove old trainings, when starting a new one | TRUE/FALSE/0/1              | FALSE   | No                           |

# Detector (Object detection)

## Images

Detector Images are published on https://hub.docker.com/r/zauberzeug/yolov5-detector.
There are two variants of the detector:

- to be deployed on a regular Linux computer, e.g. running Ubuntu (referred to as cloud-detectors)
- to be deployed on a Jetson Nano running Linux4Tegra (L4T)

Mandatory parameters are those described in [Zauberzeug Learning Loop Node Library](https://github.com/zauberzeug/learning_loop_node).
Besides, the following parameters may be set:

| Name           | Purpose                                   | Value                     | Default | Required only with ./docker.sh |
| -------------- | ----------------------------------------- | ------------------------- | ------- | ------------------------------ |
| LINKLL         | Link the node library into the container? | TRUE or FALSE             | FALSE   | Yes                            |
| DETECTOR_NAME  | Will be the name of the container         | String                    | -       | Yes                            |
| WEIGHT_TYPE    | Data type to convert weights to           | String [FP32, FP16, INT8] | FP16    | No                             |
| IOU_THRESHOLD  | IoU threshold for NMS                     | Float                     | 0.45    | No                             |
| CONF_THRESHOLD | Confidence threshold for NMS              | Float                     | 0.2     | No                             |

### Cloud-Detector

New images can be pulled with `docker pull zauberzeug/yolov5-detector:nlvX.Y.Z-cloud`, where `X.Y.Z` is the version of the node-lib used.
Legacy images can be pulled with `docker pull zauberzeug/yolov5-detector:cloud`.

Pulled images can be run with the `docker.sh` script by calling `./docker.sh run-image`.
Local builds can be run with `./docker.sh run`.
If the container does not use the GPU, try `./docker.sh d`.

### L4T-Detector

New images will be published to `docker pull zauberzeug/yolov5-detector:nlvX.Y.Z-A.B.C`, where `X.Y.Z` is the version of the node-lib used and `A.B.C` is the L4T version. Right now, the newer detector images DO NOT SUPPORT L4T.

Legacy images can be pulled with `docker pull zauberzeug/yolov5-detector:32.6.1`, where `32.6.1` is the used `Tag` (see https://hub.docker.com/r/zauberzeug/yolov5-detector/tags). It corresponds to the L4T version. Right now, `32.6.1` and `32.5.0` are supported.

# Detector (Classification)

This variant is currently in a separate subfolder yolov5_node/detector_cla. This detector is not maintained at the moment. However, the last images should work on a Linux PC.

# Publish a new release

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
The wts formats may be used by a detector running on a NVIDIA Jetson device to create an engine file as required by tensorrtx (see https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5).

# License

This code is licensed under the [AGPL-3.0 License](https://opensource.org/license/agpl-v3/). The code in

- `trainer/app_code/yolov5`
- `trainer/app_code/train_cla.py`
- `trainer/app_code/train_det.py`
- `trainer/app_code/pred_cla.py`
- `trainer/app_code/pred_det.py`
- `detector_cla/app_code/yolov5`

is largely based on the repository https://github.com/ultralytics/yolov5 which is also published under the [AGPL-3.0 License] for non-commercial use.

### Original license disclaimer in https://github.com/ultralytics/yolov5:

Ultralytics offers two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/licenses/) open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) file for more details.
- **Enterprise License**: Designed for commercial use, this license permits seamless integration of Ultralytics software and AI models into commercial goods and services, bypassing the open-source requirements of AGPL-3.0. If your scenario involves embedding our solutions into a commercial offering, reach out through [Ultralytics Licensing](https://ultralytics.com/license).
