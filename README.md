# sia-cnn
Un CNN en python con arquitectura modular

### Dependencias
- `numpy`
- `matplotlib`

### Ejecucion
Para correr el programa principal
```bash
python main.py
```

Para correr algun test
```bash
python -m tests.test
```

Para correr algun experimento
```bash
python -m experiments.experiment
```

Para que el programa funcione se debe bajar el dataset de figuras y ponerlo en una carpeta `data` en el root del proyecto
Todos los resultados se guardaran en una carpeta `results` en el root del proyecto

### Configuracion
La configuracion del `main.py` tiene la siguiente forma

```json
{
  "epochs": 10,
  "fully_connected_activation": {
    "type": "sigmoid",
    "beta": 1.0
  },
  "optimizer": {
    "type": "adam",
    "eta": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-08
  }
}
```
`epochs` indica cuantas epocas correr
`type` de la funcion de activacion puede ser: `sigmoid`, `tanh` o `relu`

Las configuraciones para los optimizadores son:

#### Gradient descent
```json
{
    "type": "gradient_descent",
    "eta": 0.01
}
```
#### Momentum
```json
{
    "type": "momentum",
    "eta": 0.01,
    "momentum": 0.9
}
```
#### ETA adaptativo
```json
{
    "type": "adaptive_eta",
    "eta": 0.01,
    "decay_factor": 0.1,
    "increase_factor": 0.1,
    "threshold": 0.01
}
```
#### ADAM
```json
{
    "type": "adam",
    "eta": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-08
  }
```

### Referencias
- [CNNs, Part 1: An Introduction to Convolution Neural Networks](https://victorzhou.com/blog/intro-to-cnns-part-1/)
- [CNNs, Part 2: Training a Convolutional Neural Network](https://victorzhou.com/blog/intro-to-cnns-part-2/)
- [cnn-from-scratch repo](https://github.com/vzhou842/cnn-from-scratch/tree/forward-only)
- [Fast convolution](https://medium.com/@thepyprogrammer/2d-image-convolution-with-numpy-with-a-handmade-sliding-window-view-946c4acb98b4)

### Dataset
- [Shapes](https://www.kaggle.com/datasets/smeschke/four-shapes)
