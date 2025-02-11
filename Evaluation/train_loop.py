import ray
import os
import subprocess

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Detect number of GPUs
NUM_GPUS = int(ray.available_resources().get("GPU", 0))
if NUM_GPUS == 0:
    raise RuntimeError("No GPUs detected by Ray.")

print(f"Detected {NUM_GPUS} GPUs.")

# Define models and datasets
MODELS = ["shufflenetv2", "googlenet", "vgg16", "mobilenet", "mobilenetv2"]
DATASETS = range(160)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

@ray.remote(num_gpus=1/3)  # Run 3 jobs per GPU
def train_model(model, dataset_num):
    """Train a model on a specific dataset using an assigned GPU."""
    
    # Get assigned GPU IDs
    gpu_ids = ray.get_gpu_ids()
    if not gpu_ids:
        print(f"[ERROR] No GPU assigned for {model}, dataset {dataset_num}")
        return

    gpu_id = gpu_ids[0]  # Assign the first available GPU

    # Set CUDA_VISIBLE_DEVICES inside subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = f"logs/{model}_dataset{dataset_num}_gpu{gpu_id}.log"
    command = ["python3", "train.py", "-net", model, "-b", "256", "-d", str(dataset_num), "-gpu"]

    # Run training subprocess
    with open(log_file, "w") as f:
        process = subprocess.Popen(command, env=env, stdout=f, stderr=f)
        return_code = process.wait()

    # Log failures
    if return_code != 0:
        with open(log_file, "a") as f:
            f.write(f"\n[ERROR] Training failed for {model}, dataset {dataset_num}, GPU {gpu_id}\n")

# Set max concurrent tasks (3 jobs per GPU)
MAX_CONCURRENT_TASKS = int(NUM_GPUS * 3)

# Launch tasks
tasks = [train_model.remote(model, dataset) for model in MODELS for dataset in DATASETS]

# Process tasks in batches
remaining_tasks = tasks
while remaining_tasks:
    num_ready = min(MAX_CONCURRENT_TASKS, len(remaining_tasks))
    done_tasks, remaining_tasks = ray.wait(remaining_tasks, num_returns=num_ready)

print("✅ All training tasks completed.")
