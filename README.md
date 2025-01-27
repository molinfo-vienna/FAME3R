FAME3R

A re-implementation of the FAME.AL model.


### Installation

1. Clone the repository and cd into the repository root:

```git clone https://github.com/molinfo-vienna/FAME3R.git```

```cd FAME3R```

2. Create a conda environment with the required python version:

```conda create --name fame3r-env python=3.10```

3. Activate the environment:

```conda activate fame3r-env```

4. Install package:

```pip install -e .```

### Usage

#### Training a model

```python scripts/train.py -i INPUT_FILE -o OUTPUT_FOLDER -r RADIUS[OPTIONAL, DEFAULT=5]```

#### Applying a trained model on some (labeled) test data

```python scripts/test.py -i INPUT_FILE -m MODEL_FILE -o OUTPUT_FOLDER -r RADIUS[OPTIONAL, DEFAULT=5]```

#### Computing the SoMs of some unlabeled data

```python scripts/infer.py -i INPUT_FILE -m MODEL_FILE -o OUTPUT_FOLDER -r RADIUS[OPTIONAL, DEFAULT=5]```
