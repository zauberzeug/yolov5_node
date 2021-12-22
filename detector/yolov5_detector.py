class Yolov5Detector:

    def __init__(self) -> None:
        super().__init__('yolov5_wts')

    def init(self,  model_info: ModelInformation, model_root_path: str):
        model_file = f'{model_root_path}/model.rt'
        


    def evaluate(self, image: Any) -> Detections:
        detections = Detections()

        except Exception as e:
            logging.exception('inference failed')

        return detections