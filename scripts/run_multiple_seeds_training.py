import subprocess
import sys

SEEDS = [42, 1337, 2025]

def run_command(command):
    """Helper function to run and log commands"""
    print(f"--> Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing: {command}")
        sys.exit(1)

def main():
    print(f"Starting Multi-Seed Pipeline for seeds: {SEEDS}")

    for seed in SEEDS:
        print(f"\n{'='*40}")
        print(f"   STARTING PIPELINE FOR SEED: {seed}")
        print(f"{'='*40}")
        
        base_ckpt = f"weights/mobilesam_seed_{seed}.pt"

        print(f"\n[Seed {seed}] Phase 1: Base Training...")
        cmd_train = [
            "python", "train_base.py",
            f"system.seed={seed}",
            f"train.output_ckpt={base_ckpt}",
            f"wandb.run_name=BASE_Seed{seed}" 
        ]
        run_command(cmd_train)

    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()