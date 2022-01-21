# Yolov5 Nodes

Learning Loop Trainer and Detector Node for Yolov5. Based on https://github.com/ultralytics/yolov5

# Detector

## Publish a new release

```
# build docker image
./docker.sh b

# publish docker image
./docker.sh p
```

# Formats

The trainer uses the `yolov5_pytorch` format identifyer.
When it saves a model to the Learning Loop it saves the model as `yolov5_pytorch` and `yolov5_wts`.
The latter is used by the detector to create an engine file as required by tensorrtx (see https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5).
