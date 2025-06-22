# -IMAGE-CLASSIFICATION-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: H LIKITHA

*INTERN ID*: CTO6DF2216

*DOMAIN*: MACHINE LEARNING

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTOSH

Within the machine learning arena, the provided code shows how to fully design an image classification system utilizing a Convolutional Neural Network (CNN).  The Python programming language is used to run the code on a Jupyter Notebook platform, and it makes extensive use of the TensorFlow framework, namely its Keras API.  Building a working CNN model that can reliably classify photos into several categories and assess its performance on a test dataset is the goal of this challenge.  CIFAR-10, a popular benchmarking dataset in the image recognition space, is the dataset used here.  It has 60,000 color photos, each measuring 32 by 32 pixels, and is categorized into 10 different classes, including cars, trucks, airplanes, birds, cats, deer, dogs, frogs, and horses. For assessing image classification models, particularly those created with deep learning methods like CNNs, this makes it the perfect option.

 All of the necessary libraries are imported at the start of the code.  These include Matplotlib for data visualization, NumPy for numerical computations, and TensorFlow and Keras for neural network construction and training.  A built-in Keras function is then used to load the CIFAR-10 dataset.  As a common preparation step in deep learning, the photos are normalized after loading by dividing each pixel value by 255 to guarantee that the data values fall between 0 and 1.  When utilizing SparseCategoricalCrossentropy, Keras automatically handles sparse label inputs, so the labels remain as integers. To give a fast overview of the variety and sorts of photos in the dataset, a sample of it is visualized using Matplotlib.  Knowing what the model will be trained on is made easier with this stage.  The CNN architecture is then built after this.  Three convolutional layers with progressively larger filter sizes and activation functions (ReLU) to add non-linearity make up the model.  A max-pooling layer, which comes after each convolutional layer, shrinks the spatial dimensions of the feature maps, assisting the model in concentrating on the most noticeable features and increasing computational efficiency. Following the last convolutional layer, the data is compressed and transferred to an output layer with 10 neurons, which correspond to the 10 classes in CIFAR-10, and then to a dense (totally connected) layer with 64 neurons.

 The Adam optimizer, which is renowned for its effectiveness and speed in deep neural network training, is used to assemble the model.  Given that the objective involves multiclass classification with integer labels, SparseCategoricalCrossentropy is the loss function that is employed.  After that, the model is trained with a batch size of 64 across 10 epochs.  Accuracy and loss are tracked during both training and validation. After training, the model is evaluated using the test dataset to measure its final performance. The expected test accuracy typically ranges between 70% and 75%, depending on the randomness of initialization and computational power. Additionally, the code includes a visualization of how the training and validation accuracy evolved across epochs, giving a clear insight into the learning behavior of the model.

In summary, this code illustrates the practical application of machine learning and deep learning in image classification. It integrates essential tools such as TensorFlow, Keras, and Matplotlib within the Jupyter Notebook platform to create a complete pipeline from loading data to evaluating model performance. This approach showcases how CNNs can be effectively used to solve real-world image classification problems using accessible and well-supported open-source libraries.

*OUTPUT:*









 
