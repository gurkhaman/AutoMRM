import ray
import os
import glob
import subprocess

ray.init(ignore_reinit_error=True)

NUM_GPUS = int(ray.available_resources().get("GPU", 0))
if NUM_GPUS == 0:
    raise RuntimeError("No GPUs detected by Ray.")
print(f"Detected {NUM_GPUS} GPUs.")

MODELS = ["shufflenetv2", "googlenet", "vgg16", "mobilenet", "mobilenetv2"]
DATASETS = range(160)

CHECKPOINT_BASE_PATH = "/workspaces/AutoMRM/Evaluation/checkpoint10/"

os.makedirs("test_logs", exist_ok=True)


def find_best_checkpoint(model, dataset_num):
    model_dir = os.path.join(CHECKPOINT_BASE_PATH, model)

    if not os.path.exists(model_dir):
        print(f"[ERROR] Model directory not found: {model_dir}")
        return None

    # Get all timestamped subdirectories and sort them by latest modified time
    subdirs = sorted(
        [
            os.path.join(model_dir, d)
            for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d))
        ],
        key=os.path.getmtime,
        reverse=True,
    )

    for subdir in subdirs:
        formatted_dataset_num = f"0{dataset_num}"
        print(f"[DEBUG] Formatted dataset number: {formatted_dataset_num}")
        checkpoint_pattern = os.path.join(
            subdir, f"{model}-{formatted_dataset_num}-*-*-best.pth"
        )
        print(f"[DEBUG] Looking for checkpoints with pattern: {checkpoint_pattern}")

        checkpoint_files = glob.glob(checkpoint_pattern)
        print(
            f"[DEBUG] Found checkpoint files: {checkpoint_files}"
        )  # ✅ See if it finds anything

        if checkpoint_files:
            best_checkpoint = max(
                checkpoint_files, key=os.path.getmtime
            )  # Pick the most recent one
            print(f"[DEBUG] Using checkpoint: {best_checkpoint}")
            return best_checkpoint

    print(f"[ERROR] No checkpoint found for {model}, dataset {dataset_num}")
    return None


@ray.remote(num_gpus=1 / 3)  # Run 3 tests per GPU
def test_model(model, dataset_num):
    gpu_ids = ray.get_gpu_ids()
    if not gpu_ids:
        print(f"[ERROR] No GPU assigned for {model}, dataset {dataset_num}")
        return

    gpu_id = gpu_ids[0]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    weights_path = find_best_checkpoint(model, dataset_num)
    if not weights_path:
        print(f"[ERROR] No valid checkpoint found for {model}, dataset {dataset_num}")
        return

    log_file = f"test_logs/{model}_dataset{dataset_num}_gpu{gpu_id}.log"
    command = [
        "python3",
        "test.py",
        "-net",
        model,
        "-weights",
        weights_path,
        "-b",
        "256",
        "-gpu",
    ]

    with open(log_file, "w") as f:
        process = subprocess.Popen(command, env=env, stdout=f, stderr=f)
        return_code = process.wait()

    if return_code != 0:
        with open(log_file, "a") as f:
            f.write(
                f"\n[ERROR] Testing failed for {model}, dataset {dataset_num}, GPU {gpu_id}\n"
            )


MAX_CONCURRENT_TASKS = int(NUM_GPUS * 3)
tasks = [test_model.remote(model, dataset) for model in MODELS for dataset in DATASETS]

remaining_tasks = tasks
while remaining_tasks:
    num_ready = min(MAX_CONCURRENT_TASKS, len(remaining_tasks))
    done_tasks, remaining_tasks = ray.wait(remaining_tasks, num_returns=num_ready)

print("✅ All testing tasks completed.")
