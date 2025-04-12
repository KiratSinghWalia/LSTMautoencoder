# LSTMautoencoder
# LSTM Autoencoder for Anomaly Detection

This repository contains an LSTM Autoencoder model implemented in PyTorch for anomaly detection in time series data. The model is trained on a dataset of sensor readings and is able to detect anomalies by identifying data points that deviate significantly from the learned patterns.

## Dataset

The model is trained on a dataset of sensor readings from a power plant. The dataset contains 6 features, and each row represents a timestamp. The dataset is preprocessed by scaling the features to have zero mean and unit variance.

## Model Architecture

The model consists of an encoder and a decoder. The encoder is a two-layer LSTM network that maps the input sequence to a fixed-length vector. The decoder is a two-layer LSTM network that maps the fixed-length vector back to the input sequence. The model is trained to minimize the reconstruction error between the input and output sequences.

## Training

The model is trained using the Adam optimizer with a learning rate of 1e-3. The model is trained for 5000 iterations with a batch size of 288. The training process is monitored by evaluating the model's performance on a validation set every 100 iterations.

## Anomaly Detection

Anomalies are detected by comparing the reconstruction error of each data point to a threshold. The threshold is determined by analyzing the distribution of reconstruction errors on a validation set. Data points with reconstruction errors above the threshold are classified as anomalies.

## Usage

To use the model, first load the model weights from the `model_weights.pth` file. Then, preprocess the input data by scaling the features to have zero mean and unit variance. Finally, pass the preprocessed data to the model's `predict` method to obtain the reconstruction error for each data point. Data points with reconstruction errors above the threshold are classified as anomalies.

## Results

The model achieves a high accuracy in detecting anomalies in the power plant dataset. The model is able to identify anomalies that are not easily detectable by visual inspection. The model can be used to monitor the power plant in real-time and to alert operators of potential problems.

## Future Work

Future work includes exploring different model architectures, such as using a convolutional neural network instead of an LSTM network. It also includes investigating different anomaly detection methods, such as using a one-class support vector machine.

## Contributing

Contributions to this repository are welcome. Please feel free to open an issue or submit a pull request.
