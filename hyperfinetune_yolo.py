import random
import subprocess

def random_search(num_trials=10):
    for i in range(num_trials):
        lr = random.uniform(0.0001, 0.1)
        batch = random.choice([16, 32, 64])
        wd = random.uniform(1e-5, 1e-2)

        command = f"python yolov5/classify/train.py --data yolo-dataset --epochs 30 --img 128 --batch {batch} --lr {lr} --decay {wd} --name finetune_{i}"
        subprocess.run(command, shell=True)

random_search(10)
