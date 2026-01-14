import os

from invoke.tasks import task
from invoke.context import Context

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
def train(ctx: Context, fraction: float = 1.0, batch_size: int = 32, max_epochs: int = 10, use_wandb_logger: bool = False, num_workers: int = 0) -> None:
    """Train model."""
    wandb_flag = "--use-wandb-logger" if use_wandb_logger else "--no-use-wandb-logger"
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py --fraction {fraction} --batch-size {batch_size} --max-epochs {max_epochs} {wandb_flag} --num-workers {num_workers}", echo=True, pty=not WINDOWS)

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
def evaluate(ctx: Context, checkpoint: str = "models/best-*.ckpt", fraction: float = 1.0, batch_size: int = BATCH_SIZE, num_workers: int = 0, data_path: str = "data/") -> None:
    """Evaluate model."""
    # find the latest checkpoint file
    import glob
    checkpoint_files = glob.glob(checkpoint)
    if not checkpoint_files:
        print("No checkpoint files found.")
        return
    checkpoint = max(checkpoint_files, key=os.path.getctime)
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py {checkpoint} --fraction {fraction} --batch-size {batch_size} --num-workers {num_workers}  --data-path {data_path}", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
