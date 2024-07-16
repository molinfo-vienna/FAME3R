FAMEAL

A re-implementation of the FAME.AL project.


### Installation

1. Clone the repository and cd into the repository root:

```git clone https://github.com/molinfo-vienna/FAMEAL.git```

```cd FAMEAL```

2. Create a conda environment with the required python version:

```conda env create --name fameal-env python=3.10```

3. Activate the environment:

```conda activate fameal-env```

4. Install package:

```pip install -e .```

### Usage

#### Training and evaluating a model

```python scripts/train.py -tr TRAIN_FILE -te TEST_FILE -o OUTPUT_FOLDER -r RADIUS[OPTIONAL, DEFAULT=5]```

#### Applying a trained model on some test data

```python scripts/test.py -i INPUT_FILE -m MODEL_FILE -o OUTPUT_FOLDER -r RADIUS[OPTIONAL, DEFAULT=5]```

#### Computing the SoMs of some data that is without experimentally confirmed SoMs

```python scripts/infer.py -i INPUT_FILE -m MODEL_FILE -o OUTPUT_FOLDER -r RADIUS[OPTIONAL, DEFAULT=5]```
