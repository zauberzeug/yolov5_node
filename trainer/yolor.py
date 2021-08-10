from learning_loop_node.trainer.training import Training
import subprocess
import training_data
import logging

def start(training: Training):
    training_data.create_file_structure(training)

    batch_size = 4  # batch size 1 takes already 6 GB on 1280x1280
    epochs = 10
    # from https://github.com/WongKinYiu/yolor#training
    cmd = f'python /yolor/train.py --batch-size {batch_size} --img 800 800 --data {training.training_folder}/dataset.yaml --cfg config.cfg --weights "weights.pt" --device 0 --name yolor --hyp /yolor/data/hyp.scratch.1280.yaml --epochs {epochs}'
    logging.info(cmd)
    # NOTE we have to write the pid inside the bash command to get the correct pid.
    cmd = f'cd {training.training_folder};{cmd} >> last_training.log 2>&1 & echo $! > last_training.pid'
    p = subprocess.Popen(cmd, shell=True)
    _, err = p.communicate()
    if p.returncode != 0:
        raise Exception(f'Failed to start training with error: {err}')
