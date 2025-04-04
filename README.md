FAME3R: a re-implementation of the FAME3 model.


### Installation

1. Create a conda environment with the required python version:

```sh
conda create --name fame3r-env python=3.10
```

2. Activate the environment:

```sh
conda activate fame3r-env
```

3. Install package:

```sh
pip install fame3r
```

### Usage

#### Determining the optimal hyperparameters via k-fold cross-validation

```sh
fame3r-cv-hp-search -i INPUT_FILE -o OUTPUT_FOLDER -r RADIUS[OPTIONAL, DEFAULT=5] -n NUMFOLDS[OPTIONAL, DEFAULT=10]
```

#### Training a model

```sh
fame3r-train -i INPUT_FILE -o OUTPUT_FOLDER -r RADIUS[OPTIONAL, DEFAULT=5]
```

#### Applying a trained model on some (labeled) test data

```sh
fame3r-test -i INPUT_FILE -m MODEL_FOLDER -o OUTPUT_FOLDER -r RADIUS[OPTIONAL, DEFAULT=5] -t THRESHOLD[OPTIONAL, DEFAULT=0.2]
```

#### Computing the SoMs of some unlabeled data

```sh
fame3r-infer -i INPUT_FILE -m MODEL_FOLDER -o OUTPUT_FOLDER -r RADIUS[OPTIONAL, DEFAULT=5] -t THRESHOLD[OPTIONAL, DEFAULT=0.2]
```
