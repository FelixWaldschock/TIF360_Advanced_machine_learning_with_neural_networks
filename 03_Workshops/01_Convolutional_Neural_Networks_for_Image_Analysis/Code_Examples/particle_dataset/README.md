# Particle Dataset

Welcome to the GitHub page of DeepTrackAI's Particle dataset. The Particle dataset is a collection of movies of optically-trapped particles used for training and evaluating deep learning models.

## Description

The Particle dataset contains 2 movies in avi format. Each frame is an RGB picture of a trapped spherical particle.

- **Dataset Size**: 2 movies
- **Number of Frames**: 100 frames per movie
- **Frame Size**: 120x120 pixels
- **Color**: RGB

## Usage

To use the Particle dataset in your project:

1. Clone this repository to your local machine.
2. Import the dataset into your machine learning framework of choice.
3. Train or evaluate your models using the dataset.

### Download via Command Line

To clone the repository and access the Particle dataset:

```bash
git clone https://github.com/DeepTrackAI/particle_dataset
cd particle_dataset
```

### Download Programmatically in Python

If you want to load the dataset directly into a Python script or Jupyter notebook:

```python
import requests
from io import BytesIO
from zipfile import ZipFile

# URL to the repository (modify this if the dataset is hosted in a specific location or file)
DATASET_URL = 'https://github.com/DeepTrackAI/particle_dataset/raw/main/mnist.zip'

response = requests.get(DATASET_URL)
with ZipFile(BytesIO(response.content)) as z:
    z.extractall()

# Now you can load the dataset using your preferred library, e.g., deeplay, PyTorch, TensorFlow.
```

## Acknowledgements

The Particle dataset was originally created by Saga Helgadottir, Aykut Argun & Giovanni Volpe.

If you use this dataset, please cite:

<https://doi.org/10.1364/OPTICA.6.000506>:
```
Saga Helgadottir, Aykut Argun, and Giovanni Volpe.
"Digital video microscopy enhanced by deep learning."
Optica 6.4 (2019): 506-513.
```

```
Benjamin Midtvedt, Saga Helgadottir, Aykut Argun, Jesús Pineda, Daniel Midtvedt, Giovanni Volpe.
"Quantitative Digital Microscopy with Deep Learning."
Applied Physics Reviews 8 (2021), 011310.
https://doi.org/10.1063/5.0034891
```

## License

The Particle dataset is made available under the terms of the [Creative Commons Attribution-Share Alike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).

## Contributing

If you find any issues with the dataset or have suggestions for improvements, please open an issue or submit a pull request.
