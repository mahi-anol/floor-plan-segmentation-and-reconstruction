import subprocess

scripts = [
    "src/pipelines/training_pipeline/unet_plus_plus_trainer.py",
    "src/pipelines/training_pipeline/novel_model_trainer.py",
    "src/pipelines/training_pipeline/Macunet_trainer.py",
    "src/pipelines/training_pipeline/Mufp_trainer.py"
]

for script in scripts:
    print(f"\nRunning {script}...\n")
    result = subprocess.run(["python", script])
    
    if result.returncode != 0:
        print(f"{script} failed. Stopping pipeline.")
        break

print("Pipeline finished.")