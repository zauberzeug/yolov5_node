from learning_loop_node.trainer.training import Training
import subprocess
import training_data


def start(training: Training):
    training_data.export(training)

    # from https://github.com/WongKinYiu/yolor#training
    cmd = f'python /yolor/train.py --batch-size 8 --img 1280 1280 --data {training.training_folder}/dataset.yaml --cfg config.cfg --weights "weights.pt" --device 0 --name yolor --hyp /yolor/data/hyp.scratch.1280.yaml --epochs 1'
    # NOTE we have to write the pid inside the bash command to get the correct pid.
    cmd = f'cd {training.training_folder};{cmd}'
    print(cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True)
    _, err = p.communicate()
    if p.returncode != 0:
        raise Exception(f'Failed to start training with error: {err}')
