# Examples


A number of simple examples are provided in the "Examples" directory.

- "simple_mnist.py" - Light-weight demo of `SimpleSharpener`,
  `Spiking_BRelu`, and `Softmax_Decode` for a fully connected net on
  mnist.

- "scheduled_mnist.py" - Convolutional net trained on mnist using
  the `ScheduledSharpener`. Uses batch normalization layers during
  training, which are removed in the final product. Should achieve
  99%+ accuracy.
