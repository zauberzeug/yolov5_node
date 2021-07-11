from coco import create_dataset
from learning_loop_node.trainer.training import Training
from icecream import ic


def test_creating_coco_from_training_object(use_test_dir):
    training = Training.parse_file('example_training.json') 
    coco = create_dataset(training).coco()

    #print(json.dumps(coco, indent=2), flush=True)
    assert len(coco['categories']) == 2
    assert len(coco['images']) == 4
    assert len(coco['annotations']) == 4*3
