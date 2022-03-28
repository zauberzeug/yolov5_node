# Yolov5 Nodes

Learning Loop Trainer and Detector Node for Yolov5. Based on https://github.com/ultralytics/yolov5

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

The trainer uses the `yolov5_pytorch` format identifyer.
When it saves a model to the Learning Loop it saves the model as `yolov5_pytorch` and `yolov5_wts`.
The latter is used by the detector to create an engine file as required by tensorrtx (see https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5).
