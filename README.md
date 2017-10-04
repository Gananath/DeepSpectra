***Warning**: This experiment is not a complete project. Because of lack of data both training and validation accuracy is less*.
# Introduction
I used to work with scientific spectral data and thought of applying machine learning in these data. In this project I have taken powder xrd specral data to classify data into seven crystal systems. The powder xrd spectrum has two main sequence a 2theta angle in *x* axis and intensity in *y* axis. I converted this sequence data into a tensor data to pass it as an image format in a convolution model. This classification is similar to the other image classification except instead of passing colour channels as depth here we are passing features (eg: 2theta,intensity). The possible application is classifiction of uv,ir,raman spectra or protein/nucleotide analysis.

![sequence2tensor](/img.jpg?raw=true "Sequence to tensor")

The idea is to find spatial or temporal features (if any) from the sequence data and then classify it into different classes.

# Requirements
- Python 2.7
- Keras with TensorFlow  backend
- Pandas

*In future I may try to add other techniques to classify this data*
