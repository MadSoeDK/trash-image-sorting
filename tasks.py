import os

from invoke.tasks import task
from invoke.context import Context
from dotenv import load_dotenv
load_dotenv()

WINDOWS = os.name == "nt"
PROJECT_NAME = "trashsorting"
PYTHON_VERSION = "3.12"
BATCH_SIZE = 32
MAX_EPOCHS = 10

# Project commands
@task
def preprocess(ctx: Context, fraction: float = 1.0) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/ --fraction {fraction}", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

@task
def docker_train(ctx: Context, fraction: float = 1.0, batch_size: int = 32, max_epochs: int = 10, use_wandb_logger: bool = True, num_workers: int = 4) -> None:
    """Train model inside docker."""
    wandb_flag = "--use-wandb-logger" if use_wandb_logger else "--no-use-wandb-logger"
    ctx.run(
        f"docker run --rm -v {os.getcwd()}:/app train:latest python -m trashsorting.train --fraction {fraction} --batch-size {batch_size} --max-epochs {max_epochs} {wandb_flag} --num-workers {num_workers}",
        echo=True,
        pty=not WINDOWS
    )

@task
def evaluate(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

@task
def predict(ctx: Context, image_path: str, checkpoint: str = "models/best-*.ckpt", verbose: bool = True) -> None:
    """Predict image class."""
    verbosity_flag = "--verbose" if verbose else ""
    # Find checkpoint file
    import glob
    ckpt_files = glob.glob(checkpoint)
    if not ckpt_files:
        print(f"No checkpoint files found matching pattern: {checkpoint}")
        return
    checkpoint = ckpt_files[0]  # Use the first matching file
    ctx.run(f"uv run src/{PROJECT_NAME}/predict.py {image_path} {checkpoint} {verbosity_flag}", echo=True, pty=not WINDOWS)

@task
def docker_train_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

@task
def docker_build_api(ctx: Context, progress: str = "plain") -> None:
    """Build API docker image."""
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

@task
def docker_run_api(ctx: Context, port: int = 8000) -> None:
    """Run API docker container."""
    ctx.run("docker rm -f trashsorting || true", echo=True, pty=not WINDOWS)
    ctx.run(
        f"docker run -d -p {port}:8000 --name trashsorting api:latest",
        echo=True,
        pty=not WINDOWS
    )

@task
def gcp_build_api(ctx: Context, progress: str = "plain") -> None:
    """Build and tag API image for GCP."""
    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION", "europe-west1")
    repo = os.getenv("GCP_REPO", "trashclassification")
    if not project_id:
        print("PROJECT_ID not set in environment variables.")
        return

    ctx.run(
        f"sudo docker build -t {region}-docker.pkg.dev/{project_id}/{repo}/api:latest "
        f"-f dockerfiles/api.dockerfile --progress={progress} .",
        echo=True,
        pty=not WINDOWS
    )

@task
def gcp_push_api(ctx: Context) -> None:
    """Push API image to Google Artifact Registry."""
    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION", "europe-west1")
    repo = os.getenv("GCP_REPO", "trashclassification")
    if not project_id:
        print("PROJECT_ID not set in environment variables.")
        return
    # Authenticate Docker with gcloud credentials
    ctx.run("cat key.json | sudo docker login -u _json_key --password-stdin https://europe-west1-docker.pkg.dev", echo=True, pty=not WINDOWS)
    ctx.run(
        f"sudo docker push {region}-docker.pkg.dev/{project_id}/{repo}/api:latest",
        echo=True,
        pty=not WINDOWS
    )

@task
def gcp_deploy_api(ctx: Context, service_name: str = "trashsorting-api") -> None:
    """Deploy API to Cloud Run."""
    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION", "europe-west1")
    repo = os.getenv("GCP_REPO", "trashclassification")
    if not project_id:
        print("PROJECT_ID not set in environment variables.")
        return

    image_url = f"{region}-docker.pkg.dev/{project_id}/{repo}/api:latest"

    ctx.run(
        f"gcloud run deploy {service_name} "
        f"--image {image_url} "
        f"--platform managed "
        f"--region {region} "
        f"--allow-unauthenticated "
        f"--project {project_id}",
        echo=True,
        pty=not WINDOWS
    )

@task
def gcp_build_train(ctx: Context, progress: str = "plain") -> None:
    """Build and tag training image for GCP."""
    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION", "europe-west1")
    repo = os.getenv("GCP_REPO", "trashclassification")
    if not project_id:
        print("PROJECT_ID not set in environment variables.")
        return

    ctx.run(
        f"sudo docker build -t {region}-docker.pkg.dev/{project_id}/{repo}/train:latest "
        f"-f dockerfiles/train.dockerfile --progress={progress} .",
        echo=True,
        pty=not WINDOWS
    )

@task
def gcp_push_train(ctx: Context) -> None:
    """Push training image to Google Artifact Registry."""
    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION", "europe-west1")
    repo = os.getenv("GCP_REPO", "trashclassification")
    if not project_id:
        print("PROJECT_ID not set in environment variables.")
        return

    # Authenticate Docker with gcloud credentials
    ctx.run("cat key.json | sudo docker login -u _json_key --password-stdin https://europe-west1-docker.pkg.dev", echo=True, pty=not WINDOWS)
    ctx.run(
        f"sudo docker push {region}-docker.pkg.dev/{project_id}/{repo}/train:latest",
        echo=True,
        pty=not WINDOWS
    )

@task
def gcp_train_vertex(ctx: Context,
                     job_name: str = None,
                     bucket_name: str = None,
                     machine_type: str = "n1-standard-4",
                     accelerator: str = None) -> None:
    """Submit training job to Vertex AI."""
    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION", "europe-west1")
    repo = os.getenv("GCP_REPO", "trashclassification")
    bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")

    if not all([project_id, bucket_name]):
        print("PROJECT_ID and GCS_BUCKET_NAME must be set.")
        return

    # Generate job name if not provided
    if not job_name:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"trash_training_{timestamp}"

    image_url = f"{region}-docker.pkg.dev/{project_id}/{repo}/train:latest"

    # Build the command
    cmd = (
        f"gcloud ai custom-jobs create "
        f"--region={region} "
        f"--display-name={job_name} "
        f"--worker-pool-spec=machine-type={machine_type},"
        f"replica-count=1,"
        f"container-image-uri={image_url} "
        f"--project={project_id}"
    )

    # Add accelerator if specified (e.g., "NVIDIA_TESLA_T4,1")
    if accelerator:
        cmd += f",accelerator-type={accelerator.split(',')[0]},accelerator-count={accelerator.split(',')[1]}"

    # Add environment variables for GCS
    cmd += (
        f" --args=--set-env-vars="
        f"GCS_BUCKET_NAME={bucket_name},"
        f"GCS_DATA_PATH=data,"
        f"GCS_MODEL_PATH=models"
    )

    ctx.run(cmd, echo=True, pty=not WINDOWS)

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
