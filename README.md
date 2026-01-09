# Lightweight model for sorting trash

This repository is a group project developed for the course Machine Learning Operations (02476) at The Technical University of Denmark (DTU).

The project aims at finetuning an open source model for classification of images of trash into categories of recyclable materials. Focus will be kept on a flexible solution, that utilizes a range of MLOps technologies 

## Project Description
Our customer requests a lightweight image classification solution capable of categorizing images of trash into a fixed set of recycling categories. The solution must be able to run locally on mobile devices with limited hardware resources. Additionally, it should be easy to extend the system with new recycling categories in order to adapt to changes in recycling regulations and differences across markets.

To accommodate these requirements, we will leverage the skills acquired during the Machine Learning Operations course to develop a proof-of-concept deep learning solution that is both robust and easily adaptable. The system will take an image of a single piece of trash as input and output an estimate of the appropriate recycling category.

Initially, the recycling categories will include paper, glass, plastic, metal, cardboard, and non-recyclable trash. However, the system should be designed to easily support additional categories, such as hazardous waste, batteries, or colored versus clear glass, without requiring major architectural changes.

As a baseline, we will fine-tune a pretrained PyTorch-based image classification model from the PyTorch Image Models (timm) library, with a focus on lightweight architectures suitable for mobile deployment (e.g., MobileNet variants). The model will be trained on the TrashNet dataset hosted on Hugging Face, which contains approximately 5,000 labeled images across the initial recycling categories. The solution will be designed to accommodate new training data and enable experimentation with different base models in a flexible and reproducible manner.

In our implementation, we will focus on utilizing tools for stable cloud-deployment such as Docker and FastAPI. Furthermore we will make use of tools for continuous integration (CI) and experiment tracking in order to ensure reproducibility, maintanability and scalability.

### Resources
- [Trashnet dataset](https://huggingface.co/datasets/garythung/trashnet)
- [LINK TO MODEL/FRAMEWORK](https://huggingface.co/datasets/garythung/trashnet)



## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
