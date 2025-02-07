import subprocess
import multiprocessing
import itertools
import torch

NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS == 0:
    raise RuntimeError("CUDA not available.")


GPUS = list(range(NUM_GPUS))
print(f"Detected {NUM_GPUS} GPUs: {GPUS}")

MODELS = ["shufflenetv2", "googlenet", "vgg16", "mobilenet", "mobilenetv2"]
BATCH_SIZE = 256

def train_model(args):
    model, dataset_num, gpu_id = args
    print(f"Starting training: Model={model}, Dataset={dataset_num}, GPU={gpu_id}")

    command = [
        "python3", "train.py",
        "-net", model,
        "-b", str(BATCH_SIZE),
        "-d", str(dataset_num),
        "-gpu", str(gpu_id)
    ]

    process = subprocess.run(command, capture_output=True, text=True)

    print(process.stdout)
    if process.stderr:
        print(f"Error in Model={model}, Dataset={dataset_num}, GPU={gpu_id}: {process.stderr}")

tasks = [(model, dataset) for model in MODELS for dataset in range(160)]
gpu_assignments = itertools.cycle(GPUS) 
task_list = [(model, dataset, next(gpu_assignments)) for model, dataset in tasks]

pool = multiprocessing.Pool(processes=NUM_GPUS)
pool.map(train_model, task_list)
pool.close()
pool.join()