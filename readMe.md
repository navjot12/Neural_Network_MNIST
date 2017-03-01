# Neural Networks on MNIST Data

This python script uses Keras based on Theano backend to train for categorization of MNIST data by application of a neural network consisting of 3 layers-  
	1. Input layer accepting digits of MNIST dataset, having shape (784,).  
	2. Hidden layer with 350 neurons.  
	3. Output layer with 10 neurons - representing the 10 output classes (digits) for MNIST dataset.  

The network has been applied on half of MNIST dataset (- a collection of 42000 handwritten digit (0-9) images) for quick computation with a quarter of this dataset used for validation. Since the dataset is too heavy to be uploaded on github (76.8 MB), it can be found at https://goo.gl/Wyl4hX.

The output of the python script can be found in the **results.txt** file. 

### Result Summary:
After 50 epochs, the training accuracy was 94.13% while validation accuracy was **92.13%**. The model seemed to have slightly overfit as an accuracy of **93.16%** was achieved after *49 epochs*.

Accuracy Progression:  
_(Blue: Training Accuracy					Red: Validation Accuracy)_
![alt text](https://github.com/navjot12/Neural_Network_MNIST/blob/master/accuracy.png "Accuracy")

Loss Progression:  
_(Blue: Training Loss						Red: Validation Loss)_
![alt text](https://github.com/navjot12/Neural_Network_MNIST/blob/master/loss.png "Loss")
