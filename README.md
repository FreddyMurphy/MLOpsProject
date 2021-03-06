MLOpsProject
==============================

A project for the Machine Learning Operations course based around a Super Resolution model.

How To Run
------------
Use ``make help`` to see how to run important features with descriptions.

Project Organization
------------

    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make train`
    ├── README.md               <- The top-level README for developers using this project.
    ├── azure                   <- Contains scipts for deploying/training models using Microsoft Azure.
    ├── data
    │   ├── external            <- Data from third party sources.
    │   ├── interim             <- Intermediate data that has been transformed.
    │   ├── processed           <- The final, canonical data sets for modeling.
    │   └── raw                 <- The original, immutable data dump.
    │
    ├── docs                    <- A default Sphinx project; see sphinx-doc.org for details. CURRENTLY NOT IN USE.
    │
    ├── models                  <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                              the creator's initials, and a short `-` delimited description, e.g.
    │                              `1.0-jqp-initial-data-exploration`.
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures             <- Generated graphics and figures to be used in reporting.
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                              generated with `pip freeze > requirements.txt`.
    ├── requirements_test.txt   <- The requirements file for running the tests.
    ├── setup.py                <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   │
    │   ├── data                <- Scripts to download or generate data
    │   ├── hparams             <- .yaml files for hyperparameter configuration using Hydra.
    │   ├── models              <- Scripts to train models and then use trained models to make
    │                              predictions
    │── tests                   <- Test scripts using pytest.

--------

Project Checklist
------------
The following checklist gives a good sense of what is included in the project:
### Week 1

- [x] ~~Create a git repository~~
- [x] ~~All members have write access to repository~~
- [x] ~~Using dedicated environment to keep track of packages~~
- [x] ~~File structure made using cookiecutter~~
- [x] ~~make_dataset.py filled to download needed data~~
- [x] ~~Add a model file and a training script and get that running~~
- [x] ~~Done profiling and optimized code~~
- [x] ~~requirements.txt filled with used dependencies~~
- [x] ~~Write unit tests for some part of the codebase and get code coverage~~
- [x] ~~Get some continues integration running on the github repository~~
- [x] ~~use either tensorboard or wandb to log training progress and other important metrics/artifacts in your code~~
- [x] ~~remember to comply with good coding practices while doing the project~~

### Week 2

- [x] ~~Setup and used Azure to train your model~~
- [x] ~~Played around with distributed data loading~~
- [x] ~~(not curriculum) Reformated your code in the pytorch lightning format~~
- [x] ~~Deployed your model using Azure~~
- [ ] ~~Checked how robust your model is towards data drifting~~
- [ ] ~~Deployed your model locally using TorchServe~~

### Week 3

- [x] ~~Used Optuna to run hyperparameter optimization on your model~~
- [x] ~~Wrote one or multiple configurations files for your experiments~~
- [x] ~~Used Hydra to load the configurations and manage your hyperparameters~~

### Additional

- [x] ~~Revisit your initial project description. Did the project turn out as you wanted?~~
- [x] ~~Make sure all group members have a understanding about all parts of the project~~
- [x] ~~Created a powerpoint presentation explaining your project~~
- [x] ~~Uploaded all your code to github~~

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>