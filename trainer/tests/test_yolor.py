from learning_loop_node.trainer.training import Training
import yolor


def test_start_training(use_test_dir):
    yolor.start(Training.parse_file('example_training.json'))
