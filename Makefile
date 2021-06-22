.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

export PATH := bin:$(PATH)

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = MLOpsProject
OS := $(shell uname)

ifeq ($(OS),Darwin)
PYTHON_INTERPRETER = python3
else
PYTHON_INTERPRETER = python
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
requirements: test_environment
	$(shell touch wandb_api_key.txt)
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -U -r requirements.txt

## Install Python test dependencies
test_requirements:
	$(PYTHON_INTERPRETER) -m pip install -U -r requirements_tests.txt

## Download and process the DIV2K dataset
data: requirements
	$(shell mkdir -p data)
	$(shell mkdir -p data/raw)
	$(shell mkdir -p data/processed)
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Train model using DIV2K locally
train:
	$(PYTHON_INTERPRETER) src/models/main.py session=train

## Optimize hyperparameters using Optuna
train-optuna:
	$(PYTHON_INTERPRETER) src/models/main.py session=train --multirun

## Train model using DIV2K using Azure
train-azure:
	$(PYTHON_INTERPRETER) azure/run_config.py

## Evaluate a given model using DIV2K
evaluate:
	$(PYTHON_INTERPRETER) src/models/main.py session=evaluate

## Deploy most recently trained model using Azure
deploy:
	$(PYTHON_INTERPRETER) azure/deploy_model.py

## Delete all currently deployed services on Azure
delete:
	$(PYTHON_INTERPRETER) azure/delete_services.py

## Run tests
test: test_requirements
	pytest -v

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
