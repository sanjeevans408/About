NEURAL NETWORKS AND DEEP LEARNING LAB MANUAL (CCS355)

LIST OF LAB EXPERIMENTS:
1.	Implement simple vector addition in TensorFlow.
2.	Implement a regression model in Keras.
3.	Implement a perceptron in TensorFlow/Keras Environment.
4.	Implement a Feed-Forward Network in TensorFlow/Keras.
5.	Implement an Image Classifier using CNN in TensorFlow/Keras.
6.	Improve the Deep learning model by fine tuning hyper parameters.
7.	Implement a Transfer Learning concept in Image Classification.
8.	Using a pretrained model on Keras for Transfer Learning
9.	Perform Sentiment Analysis using RNN
10.	Implement an LSTM based Auto encoder in Tensor Flow/Keras.
11.	Image generation using GAN































Ex No. 1	Implement Simple Vector Addition In Tensor Flow

Aim:
This program defines two constant tensors vector1 and vector2, representing the vectors [1, 2, 3] and [4, 5, 6] respectively. Then, it performs addition using tf.add() and prints the result.
Program:
import tensorflow as tf
# Define the vectors
vector1 = tf.constant([1, 2, 3])
vector2 = tf.constant([4, 5, 6])
# Perform vector addition
result = tf.add(vector1, vector2)
# Print the result directly
print("Result of vector addition:", result.numpy())

Output:
Result of vector addition: [5 7 9]











Result:
Thus the Program to Implement Simple Vector Addition In Tensor Flow
has been executed successfully.

Ex No. 2	Implement A Regression Model in keras


Aim:
The aim of this exercise is to demonstrate how to implement a simple regression model using the Keras API, a high-level neural networks library running on top of TensorFlow.
Algorithm:
1. Generate Or Load The Dataset.
2. Define A Sequential Model.
3. Add Layers To The Model. For Regression, Typically A Single Dense Layer Is
Sufficient.
4. Compile The Model, Specifying Optimizer And Loss Function.
5. Train The Model On The Dataset Using The Fit Method.
6. Make Predictions Using The Trained Model.
7. Evaluate The Model's Performance Using Appropriate Metrics.
Program:
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Generate some random data for regression
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1  # y = 2X + 1 + noise
# Define the regression model using Keras
model = keras.Sequential([layers.Dense(1, input_shape=(1,))])
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(X, y, epochs=100, verbose=1)
# Predict using the trained model
X_test = np.array([[0.1], [0.2], [0.3]])
predictions = model.predict(X_test)
print("Predictions:")
print(predictions)

Output:
Epoch 1/100
4/4 [==============================] - 0s 749us/step - loss: 4.5494
Epoch 2/100
4/4 [==============================] - 0s 623us/step - loss: 4.4588
...
Epoch 100/100
4/4 [==============================] - 0s 499us/step - loss: 0.0345
Predictions:
[[1.2399716]
[1.4402403]
[1.640509 ]]

















Result:
Thus the Program to Implement a Regression Model in keras
has been executed successfully.

Ex No. 3	Implement A Perceptron In Tensor Flow And Keras

Aim:
The aim of this exercise is to implement a perceptron, a fundamental building block of neural networks, in both TensorFlow and Keras environments. Through this exercise, will gain an understanding of how to create and train a perceptron model using both of these popular deep learning frameworks.
Algorithm:
1. Initialize Weights and Bias: Initialize weights and bias randomly or with predefined
values.
2. Define Input Data: Prepare the input data along with corresponding labels for training the perceptron.
3. Define Placeholder (TensorFlow) or Input Shape (Keras): Depending on the framework,define placeholders for input data (TensorFlow) or specify the input shape (Keras).
4. Define Perceptron Operation: Compute the output of the perceptron by performing the weighted sum of inputs and adding a bias term, followed by passing the result through an activation function (e.g., step function, sigmoid, etc.).
5. Define Loss Function: Define a suitable loss function to measure the discrepancy between predicted and actual outputs (e.g., mean squared error, binary cross-entropy).
6. Define Optimizer: Choose an optimizer algorithm (e.g., stochastic gradient descent,
Adam) to minimize the loss and update the weights and bias accordingly.
7. Initialize Variables (TensorFlow): Initialize all variables used in the computation graph.
8. Train the Perceptron: Iterate over the training data multiple times, feeding the input data and labels to the perceptron, and optimizing the weights and bias using the chosen optimizer.
9. Evaluate the Perceptron (Optional): If necessary, evaluate the trained perceptron on a separate validation or test dataset to assess its performance.
10. Make Predictions: Use the trained perceptron to make predictions on new, unseen data.
Program:
Tensor Flow:
import numpy as np
import tensorflow as tf
# Define the input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])
# Define the perceptron
class Perceptron(tf.Module):
    def init (self):
        super(Perceptron, self). init ()
        self.W = tf.Variable(tf.random.uniform([2, 1]))
        self.b = tf.Variable(tf.random.uniform([1]))
    def call (self, x):
        return tf.math.sigmoid(tf.matmul(x, self.W) + self.b)
    # Instantiate the perceptron
        perceptron = Perceptron()
    # Define the loss function and optimizer
        loss_function = tf.losses.BinaryCrossentropy()
        optimizer = tf.optimizers.Adam(learning_rate=0.1)
    # Training loop
        for epoch in range(1000):
            with tf.GradientTape() as tape:
                predictions = perceptron(X)
                loss = loss_function(y, predictions)
                gradients = tape.gradient(loss, perceptron.trainable_variables)
                optimizer.apply_gradients(zip(gradients, perceptron.trainable_variables))
    # Test the perceptron
                test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                predictions = perceptron(test_data)
                print("Predictions:")
                print(predictions.numpy())
keras:
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Define the input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])
# Define the perceptron model
model = Sequential([
Dense(1, input_dim=2, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')
# Train the model
model.fit(X, y, epochs=1000, verbose=0)
# Test the model
predictions = model.predict(X)
print("Predictions:")
print(predictions)

OUTPUT:
Predictions:
[[0.06273559]
[0.16103294]
[0.10974989]
[0.8888136 ]]



















Result:
Thus the Program to Implement A Perceptron In Tensor Flow And Keras
has been executed successfully.
Ex No. 4	Implement A Feed Forward Network In Tensor Flow / Keras

Aim:
The aim of this exercise is to implement a feedforward neural network, also known as a multilayer perceptron (MLP), in both TensorFlow and Keras environments. By completing this exercise, participants will gain an understanding of how to build and train a basic feedforward neural network using these popular deep learning frameworks.
Algorithm:
1. Initialize Weights and Biases: Initialize the weights and biases of each layer randomly or using predefined values.
2. Define Input Data: Prepare the input data along with corresponding labels for training the neural network.
3. Define Model Architecture:
a. TensorFlow:
- Define placeholders for input data.
- Define variables for weights and biases for each layer.
- Define the architecture of the neural network by specifying the number of layers, number of neurons in each layer, and activation functions.
b. Keras:
- Initialize a sequential model.
- Add layers to the model using Dense layer specifying the number of neurons and activation functions.
- Compile the model, specifying the optimizer, loss function, and metrics.
4. Forward Propagation:
a. TensorFlow:
- Implement forward propagation by applying the activation function to the
linear combination of inputs, weights, and biases for each layer.
b. Keras:
- Keras handles forward propagation internally during the training process.
5. Define Loss Function: Define a suitable loss function to measure the discrepancy between\ predicted and actual outputs (e.g., mean squared error, categorical cross-entropy).
6. Define Optimizer: Choose an optimizer algorithm (e.g., stochastic gradient descent,
Adam) to minimize the loss and update the weights and biases accordingly.
7. Initialize Variables (TensorFlow): Initialize all variables used in the computation graph.
8. Train the Neural Network: Iterate over the training data multiple times, feeding the input data and labels to the neural network, and optimizing the weights and biases using the
chosen optimizer.
9. Evaluate the Model (Optional): If necessary, evaluate the trained neural network on a separate validation or test dataset to assess its performance.
10. Make Predictions: Use the trained neural network to make predictions on new, unseendata.
Program:
Tensor flow:
import numpy as np
import tensorflow as tf
# Generate some random data for binary classification
np.random.seed(0)
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, (100, 1))
# Define the feedforward neural network
model = tf.keras.Sequential([
tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
tf.keras.layers.Dense(4, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X, y, epochs=100, verbose=0)
# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print("Loss:", loss)
print("Accuracy:", accuracy)

Keras:
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Generate some random data for binary classification
np.random.seed(0)
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, (100, 1))
# Define the feedforward neural network using Keras
model = Sequential([
Dense(4, activation='relu', input_shape=(2,)),
Dense(4, activation='relu'),
Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X, y, epochs=100, verbose=0)
# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print("Loss:", loss)
print("Accuracy:", accuracy)

Output:
Loss: 0.561991870880127
Accuracy: 0.7400000095367432


















Result:
Thus the Program to Implement A Feed Forward Network In Tensor Flow / Keras
has been executed successfully.
Ex No. 5	Implement An Image Classifier Using Cnn In Tensor Flow And Keras

Aim:
The aim of this exercise is to implement an image classifier using Convolutional Neural Networks (CNNs) in both TensorFlow and Keras environments. By completing this exercise, we will gain an understanding of how to build and train a CNN-based image classifier using these popular deep learning frameworks.
Algorithm:
1. Prepare Dataset: Load and preprocess the image dataset. This may involve resizing images to a fixed size, normalizing pixel values, and splitting the dataset into training, validation, and test sets.
2. Define Model Architecture:
a. TensorFlow:
- Define placeholders for input images and labels.
- Define the architecture of the CNN by specifying convolutional layers,
pooling layers, and fully connected layers.
b. Keras:
- Initialize a sequential model.
- Add convolutional layers using Conv2D and pooling layers using
MaxPooling2D.
- Add fully connected layers using Dense.
- Compile the model, specifying the optimizer, loss function, and metrics.
3. Data Augmentation (Optional):
4. Optionally apply data augmentation techniques such as random rotation, flipping, andshifting to increase the diversity of the training data.
5. Compile Model:
6. Define the loss function, optimizer, and evaluation metrics for training the model.
7. Train Model:
8. Feed the training data to the model and train it using the fit method. Monitor the trainingprocess using validation data.
9. Evaluate Model.
10. Evaluate the trained model on the test set to assess its performance.
11. Fine-Tuning and Hyperparameter Tuning (Optional):
12. Optionally fine-tune the model architecture and hyperparameters based on the
performance on the validation set.
13. Make Predictions
14. Use the trained model to make predictions on new, unseen images.
Program:
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64)
# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)


Output:
Epoch 1/5
938/938 [==============================] - 6s 6ms/step - loss: 0.1481 - accuracy:
0.9546
Epoch 2/5
938/938 [==============================] - 6s 6ms/step - loss: 0.0455 - accuracy:
0.9859
Epoch 3/5
938/938 [==============================] - 6s 6ms/step - loss: 0.0327 - accuracy:
0.9899
Epoch 4/5
938/938 [==============================] - 6s 6ms/step - loss: 0.0247 - accuracy:
0.9921
Epoch 5/5
938/938 [==============================] - 6s 6ms/step - loss: 0.0191 - accuracy:
0.9940
313/313 [==============================] - 1s 3ms/step - loss: 0.0254 - accuracy:
0.9916










Result:
Thus the Program to Implement An Image Classifier Using Cnn In Tensor FlowAnd Kerashas been executed successfully.

Ex No. 6	Improve Deep Learning Model By Fine Tuning Hyper Parameters

Aim:
The aim of hyperparameter tuning in deep learning is to improve the performance of amodel by systematically searching for the optimal combination of hyperparameters.Hyperparameters are settings that control the learning process, such as the learning rate, batch size,number of layers, number of neurons per layer, activation functions, etc. By fine- tuning thesehyperparameters, we aim to achieve better accuracy, faster convergence, and improvedgeneralization of the model.
Algorithm:
1. Define Hyper parameter Space:
Define the hyperparameters to be tuned and their respective search spaces. For example, thelearning rate may be searched in the range [0.0001, 0.1], the number of neurons per layermay be chosen from [32, 64, 128], etc.2. Choose Optimization Strategy:Select a hyperparameter optimization strategy. Common strategies include grid search, randomsearch, Bayesian optimization, and more advanced techniques like genetic algorithms orevolutionary strategies.
3. Split Data:Split the dataset into training, validation, and test sets. The validation set will be used toevaluate the performance of each set of hyperparameters during the tuning process.
4. Define Model Architecture:Define the architecture of the deep learning model. This includes the number of layers, types oflayers (e.g., convolutional, recurrent, dense), activation functions, regularization techniques,
etc.
5. Define Training Procedure:Define the training procedure, including the optimizer, loss function, and any additionalcallbacks or metrics to monitor during training.
6. Hyperparameter Optimization Loop:
• Start the hyperparameter optimization loop:
• Sample a set of hyperparameters from the search space.
• Build and train the model using the sampled hyperparameters on the training data.
• Evaluate the model on the validation set.
• Keep track of the validation performance for each set of hyperparameters.
• Repeat the above steps until a predefined budget or stopping criteria is reached.
7. Select Best Hyperparameters:Once the hyperparameter optimization loop is complete, select the set of hyperparameters thatachieved the best performance on the validation set.
8. Evaluate Final Model:
• Train the final model using the selected hyperparameters on the entire training dataset(training + validation).
• Evaluate the final model on the test set to obtain an unbiased estimate of its
performance.
Program:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Load dataset
data = pd.read_csv(r"S:\tony\python files\synthetic_data.csv")  # Change this to your actual dataset
# Define features and labels
X = data.drop("target_column", axis=1)  # Change 'target_column' to actual target column
y = data["target_column"]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define a function to create the deep learning model
def create_model(optimizer='adam', activation='relu', neurons=16):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
# Create KerasClassifier for GridSearchCV
model = KerasClassifier(build_fn=create_model, verbose=0)
# Define the hyperparameters grid
param_grid = {
    'model__optimizer': ['adam', 'rmsprop'],
    'model__activation': ['relu', 'tanh'],
    'model__neurons': [8, 16, 32]
}
# Perform GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Evaluate model on test data with best parameters
best_model = grid.best_estimator_
accuracy = best_model.score(X_test, y_test)
print("Test Accuracy: %.2f%%" % (accuracy * 100))

OUTPUT:
Best: 0.85 using {'activation': 'relu', 'neurons': 32, 'optimizer': 'adam'}
0.80 (0.02) with: {'activation': 'relu', 'neurons': 8, 'optimizer': 'adam'}
0.82 (0.01) with: {'activation': 'relu', 'neurons': 8, 'optimizer': 'rmsprop'}
...
Test Accuracy: 86.50%











Result
Thus the Program to Implement Deep Learning Model By Fine Tuning Hyper Parameters has been executed successfully.
Ex No. 7	Implement A Transfer Learning Concept In Image Classification

Aim:
The aim of implementing transfer learning in image classification is to leverage theknowledge learned by a pre-trained model on a large dataset (source domain) and transfer it to anew, possibly smaller dataset (target domain). Transfer learning allows us to train a model withlimited data and resources while still achieving good performance.

Algorithm:
1. Select Pre-trained Model:Choose a pre-trained model that has been trained on a large dataset with similar characteristicsto the target dataset. Common choices include VGG, ResNet, Inception, and MobileNet.
2. Load Pre-trained Model:Load the pre-trained model and remove the top layers (fully connected layers) that are responsiblefor making predictions on the original task.
3. Freeze Base Layers (Optional):Optionally, freeze the weights of the base layers (the layers of the pre-trained model) to preventthem from being updated during training. This step is recommended when the target dataset issmall and similar to the source dataset.
4. Add New Classification Layers:Add new layers (typically fully connected layers) on top of the pre-trained base layers. Theselayers will be responsible for making predictions on the new target task.
5. Define Training Procedure:Define the training procedure, including the optimizer, loss function, and any additional callbacksor metrics to monitor during training.
6. Data Augmentation (Optional):Optionally, apply data augmentation techniques to increase the diversity of the training data andimprove the model's generalization ability.
7. Train the Model:Train the model on the target dataset. Since the base layers are frozen (or partially frozen), onlythe newly added layers will be trained. This step helps the model learn task-specific featuresfrom the target dataset while leveraging the knowledge learned by the pre-trained model.
8. Fine-tuning (Optional):Optionally, unfreeze some of the base layers and continue training the entire model end-to-end.Fine-tuning allows the model to further adapt to the target dataset and potentially improveperformance.
9. Evaluate the Model:Evaluate the trained model on a separate validation set to assess its performance. Monitor metricssuch as accuracy, precision, recall, and F1-score.
10. Deploy the Model:Once satisfied with the model's performance, deploy it to make predictions on new, unseenimages from the target domain.
Program:
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
# Define constants
img_width, img_height = 224, 224
train_data_dir = 'train'
validation_data_dir = 'validation'
batch_size = 32
num_classes = 2
epochs = 10
# Preprocess and augment data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary' if num_classes == 2 else 'categorical'
)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary' if num_classes == 2 else 'categorical'
)

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False
# Create a new model on top of VGG16
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid'))
# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
    metrics=['accuracy']
)
# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
# Evaluate the model
scores = model.evaluate(validation_generator)
print("Validation Loss: %.2f%%" % (scores[0] * 100))
print("Validation Accuracy: %.2f%%" % (scores[1] * 100))






Output:
Found 2000 images belonging to 2 classes.
Found 800 images belonging to 2 classes.
Epoch 1/10
62/62 [==============================] - 125s 2s/step - loss: 0.6138 - accuracy:
0.6746 - val_loss: 0.4546 - val_accuracy: 0.8203
Epoch 2/10
62/62 [==============================] - 123s 2s/step - loss: 0.4683 - accuracy:
0.7778 - val_loss: 0.3788 - val_accuracy: 0.8320
...
Epoch 10/10
62/62 [==============================] - 122s 2s/step - loss: 0.3621 - accuracy:
0.8422 - val_loss: 0.2734 - val_accuracy: 0.8958
25/25 [==============================] - 23s 923ms/step - loss: 0.2841 - accuracy:
0.8863
Validation Loss: 28.41%
Validation Accuracy: 88.63%













Result:
Thus the Program to Implement A Transfer Learning Concept In Image Classification has been executed successfully.
Ex No. 8	Using A Pretrained Model On Keras For Transfer Learning

Aim:
The aim of using a pre-trained model for transfer learning in Keras is to leverage theknowledge learned by a pre-trained deep learning model on a large dataset and transfer it to anew task or dataset. By fine-tuning the pre-trained model on the new data, we aim to achievebetter performance with less training time and fewer data.
Algorithm:
1. Select Pre-trained Model:Choose a pre-trained model from the Keras applications module. Popular choices includeVGG16, VGG19, ResNet50, InceptionV3, Xception, and MobileNet.
2. Load Pre-trained Model:Load the pre-trained model along with its weights trained on ImageNet dataset. Excludethe top layers (fully connected layers) that are responsible for making predictions on theoriginal task.
3. Modify Model Architecture:Add new layers on top of the pre-trained model to adapt it to the new task. The newlayers will include a Global Average Pooling layer and a Dense output layer with theappropriate number of units for the new classification task.
4. Freeze Base Layers (Optional):Optionally, freeze the weights of the base layers (the layers of the pre-trained model) toprevent them from being updated during training. This step is recommended when thetarget dataset is small and similar to the source dataset.
5. Compile Model:Compile the model with an appropriate optimizer, loss function, and evaluation metric.
6. Data Augmentation (Optional):Optionally, apply data augmentation techniques to increase the diversity of the trainingdata and improve the model's generalization ability. This step is especially useful working with limited data.
7. Train the Model:Train the model on the target dataset. Since the base layers are frozen (or partially frozen),only the newly added layers will be trained. This step helps the model learn task-specificfeatures from the target dataset while leveraging the knowledge learned by the pre-trainedmodel.
8. Fine-tuning (Optional):Optionally, unfreeze some of the base layers and continue training the entire model end-toend.Fine-tuning allows the model to further adapt to the target dataset and potentiallyimprove performance.
9. Evaluate the Model:Evaluate the trained model on a separate validation set to assess its performance. Monitormetrics such as accuracy, precision, recall, and F1-score.
10. Deploy the Model:Once satisfied with the model's performance, deploy it to make predictions on new, unseendata.
Program:
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
# Define constants
img_width, img_height = 224, 224
train_data_dir = "train"
validation_data_dir = "validation"
batch_size = 32
num_classes = 2
epochs = 10
# Ensure dataset folders exist
if not os.path.exists(train_data_dir) or not os.path.exists(validation_data_dir):
    raise FileNotFoundError("Dataset folders not found. Ensure you have extracted the dataset correctly.")
# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
# Load dataset
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)
# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False
# Create new model on top of VGG16
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
# Evaluate the model
scores = model.evaluate(validation_generator)
print("Validation Loss: %.2f%%" % (scores[0] * 100))
print("Validation Accuracy: %.2f%%" % (scores[1] * 100))










Output:
Found 2000 images belonging to 2 classes. Found
800 images belonging to 2 classes. Epoch 1/10
62/62 [==============================]
- 125s 2s/step - loss: 0.6138 - accuracy: 0.6746
- val_loss: 0.4546 - val_accuracy: 0.8203
...
Epoch 10/10
62/62 [==============================]
- 122s 2s/step - loss: 0.3621 - accuracy: 0.8422 - val_loss: 0.2734 - val_accuracy: 0.8958 25/25
[==============================]
- 23s 923ms/step - loss: 0.2841 - accuracy: 0.8863
Validation Loss: 28.41%
Validation Accuracy: 88.63%
















Result:
Thus the Program to Implement A Pretrained Model On Keras For Transfer Learning has been executed successfully
Ex No. 8	Perform sentiment analysis using run Parameters

Aim:
The aim of this lab exercise is to implement a sentiment analysis model using RecurrentNeural Networks (RNNs) to classify text data into positive or negative sentiment categories.
Algorithm:
1. Dataset Preparation:
• Use a labeled dataset for sentiment analysis, such as the IMDb movie review dataset, whichconsists of movie reviews labeled as positive or negative.
• Split the dataset into training, validation, and test sets.
2. Data Preprocessing:
• Clean the text data: Remove special characters, punctuation, and unwanted symbols.
• Tokenization: Split the text into individual words or tokens.
• Convert words to indices: Map each word to a unique integer index.
• Padding: Ensure all sequences have the same length by padding shorter sequences withzeros or truncating longer sequences.
3. Model Architecture:
• Define an RNN architecture using libraries like TensorFlow or PyTorch.
• Choose the type of RNN cell (e.g., LSTM, GRU).
• Stack multiple RNN layers for better performance if needed.
• Add a Dense layer with sigmoid activation for binary classification (positive or negativesentiment).
4. Model Training:
• Initialize the RNN model.
• Compile the model with appropriate loss function (e.g., binary cross-entropy) and optimizer(e.g., Adam).
• Train the model on the training data.
• Monitor the training process using metrics like accuracy and loss on the validation set.
• Tune hyperparameters like learning rate, batch size, and number of epochs based on
validation performance.
5. Evaluation:
• Evaluate the trained model on the test set to assess its performance.
• Calculate metrics such as accuracy, precision, recall, and F1-score.
• Visualize the performance metrics and, if necessary, confusion matrices.
6. Inference:
• Use the trained model to predict sentiment on new, unseen text data.
• Preprocess the new text data similarly to the training data.
• Feed the preprocessed data into the trained model for prediction.
7. Experimentation and Improvement:
• Experiment with different model architectures, hyperparameters, and preprocessing
techniques to improve performance.
• Explore the use of pre-trained word embeddings to enhance the model's understanding oftext semantics.
• Consider advanced techniques like attention mechanisms or bidirectional RNNs for bettercapturing context.
8. Conclusion:
• Summarize the findings of the lab exercise, including the performance of the sentimentanalysis model and any insights gained during experimentation.
• Discuss potential future directions for further improvement or research.
Program:
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
print("Keras Tokenizer imported successfully!")
from sklearn.model_selection import train_test_split
# Sample data (replace this with your dataset)
texts = ["I love this movie", "This movie is great", "I hate this movie", "This movie is terrible"]
labels = [1, 1, 0, 0] # 1 for positive, 0 for negative
# Tokenize the texts
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
# Pad sequences to ensure uniform input size
maxlen = 10
data = pad_sequences(sequences, maxlen=maxlen)
# Convert labels to numpy array
labels = np.asarray(labels)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# Build the RNN model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=32, input_length=maxlen))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

















Output:
Epoch 1/10
1/1 [==============================] - 1s 1s/step - loss: 0.6927 - accuracy: 0.5000 -
val_loss: 0.6931 - val_accuracy: 0.5000
...
Epoch 10/10
1/1 [==============================] - 0s 36ms/step - loss: 0.6583 - accuracy: 1.0000 -
val_loss: 0.6963 - val_accuracy: 0.5000
1/1 [==============================] - 0s 25ms/step - loss: 0.6963 - accuracy: 0.5000
Test Loss: 0.6962565774917603
Test Accuracy: 0.5














Result:
Thus the Program to Perform sentiment analysis using run Parameters has been executed successfully
Ex No. 9	Implement An Lstm Based Auto Encoder In          TensorFlow/ Keras

Aim:
The aim of this lab exercise is to implement an LSTM-based autoencoder using
TensorFlow/Keras for sequence data compression and reconstruction.
Algorithm:
1. Dataset Preparation:
• Use a dataset containing sequential data, such as time series or text data.
• Split the dataset into training and test sets.
2. Data Preprocessing:
• Normalize or scale the data if necessary.
• Convert the sequential data into fixed-length sequences.
• Optionally, add noise to the input sequences to improve the robustness of the autoencoder.
3. Model Architecture:
• Define an LSTM-based autoencoder architecture using TensorFlow/Keras.
• Create an encoder LSTM layer to compress the input sequence into a fixed-length latentrepresentation.
• Create a decoder LSTM layer to reconstruct the input sequence from the latent
representation.
• Connect the encoder and decoder layers to create the autoencoder model.
4. Model Training:
• Initialize the LSTM autoencoder model.
• Compile the model with an appropriate loss function, such as mean squared error (MSE),and optimizer, such as Adam.
• Train the model on the training data.
• Monitor the training process and tune hyperparameters as needed.
5. Evaluation:
• Evaluate the trained autoencoder model on the test set.
• Calculate reconstruction error between the original and reconstructed sequences.
• Visualize the reconstructed sequences to assess the quality of reconstruction.
6. Application:
• Use the trained autoencoder model for tasks such as sequence denoising or anomaly
detection.
• Apply the encoder part of the model to compress sequences into a lower-dimensional
latent space for downstream tasks.
7. Experimentation and Improvement:
• Experiment with different architectures, such as adding more LSTM layers or using
bidirectional LSTMs, to improve reconstruction performance.
• Explore different loss functions and regularization techniques to enhance the stability and
generalization of the model.
• Consider incorporating attention mechanisms or other advanced techniques to improve the
autoencoder's ability to capture long-range dependencies.
8. Conclusion:
• Summarize the findings of the lab exercise, including the performance of the LSTM-based
autoencoder and any insights gained during experimentation.
• Discuss potential applications and future research directions for further exploration.
Program:
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
# Generate some random data for demonstration
data = np.random.rand(100, 10, 1) # shape: (samples, timesteps, features)
# Define the model architecture
latent_dim = 2 # Dimension of the latent space representation
# Encoder
inputs = Input(shape=(data.shape[1], data.shape[2]))
encoded = LSTM(latent_dim)(inputs)
# Decoder
decoded = RepeatVector(data.shape[1])(encoded)
decoded = LSTM(data.shape[2], return_sequences=True)(decoded)
# Autoencoder
autoencoder = Model(inputs, decoded)
# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')
# Train the model
autoencoder.fit(data, data, epochs=50, batch_size=32, validation_split=0.2)
# Test the model by reconstructing the input data
reconstructed_data = autoencoder.predict(data)
# Print example of original and reconstructed data
print("Original Data:")
print(data[0])
print("\nReconstructed Data:")
print(reconstructed_data[0])
Output:
Epoch 1/50
3/3 [==============================] - 1s 99ms/step - loss: 0.2733 - val_loss: 0.2578
Epoch 2/50
3/3 [==============================] - 0s 12ms/step - loss: 0.2670 - val_loss: 0.2520
...
Epoch 50/50
3/3 [==============================] - 0s 10ms/step - loss: 0.0485 - val_loss: 0.044
Original Data:
[[0.75828457]
[0.72486731]
[0.1619405 ]
[0.84798926]
[0.77168659]
[0.15825592]
[0.13336555]
[0.56128449]
[0.14060701]
[0.74388057]]
Reconstructed Data:
[[0.68393844]
[0.7702495 ]
[0.15474795]
[0.7850587 ]
[0.72745067]
[0.1947125 ]
[0.1536809 ]
[0.58117414]
[0.15412793]
[0.7389922 ]]
Result:
Thus the Program to Implement An Lstm Based Auto Encoder In          TensorFlow/ Kerashas been executed successfully
Ex No. 10	Image generation using GAN

Aim:
The aim of this lab exercise is to implement a Generative Adversarial Network (GAN) usingTensorFlow/Keras for generating realistic images.
Algorithm:
1. Dataset Preparation:
• Choose a dataset suitable for image generation, such as CIFAR-10, CelebA, or MNIST.
• Preprocess the dataset, including normalization and resizing, if necessary.
• Split the dataset into training and validation sets.
2. Generator Model:
• Define the generator model architecture using TensorFlow/Keras.
• Start with a simple architecture, such as a series of dense layers followed by upsampling layers(e.g., transposed convolutions or upsampling layers).
• Use activation functions like ReLU or LeakyReLU for intermediate layers and tanh for theoutput layer to ensure pixel values are in the range [-1, 1].
3. Discriminator Model:
• Define the discriminator model architecture using TensorFlow/Keras.
• Start with a convolutional neural network (CNN) architecture to classify real and generatedimages.
• Use activation functions like LeakyReLU and sigmoid for the output layer to produce aprobability score indicating the likelihood of the input image being real.
4. GAN Model:
• Combine the generator and discriminator models to form the GAN model.
• Freeze the discriminator weights during GAN training to prevent the generator from
overpowering the discriminator too early.
• Compile the GAN model with appropriate loss functions (e.g., binary cross-entropy) andoptimizer (e.g., Adam).
5. Training:
• Train the GAN model iteratively in alternating steps:
• Train the discriminator using batches of real and fake images, adjusting its weights to betterdistinguish between real and generated images.
• Train the generator by generating fake images and trying to fool the discriminator intoclassifying them as real.
• Monitor the training process and adjust hyperparameters such as learning rate and batch size asneeded.
6. Evaluation:
• Evaluate the trained GAN model on the validation set to assess the quality of generated images.
• Visualize generated images and compare them with real images to evaluate realism anddiversity.
• Calculate metrics like Inception Score or Frechet Inception Distance (FID) to quantitativelyevaluate the quality of generated images.
7. Fine-tuning and Optimization:
• Experiment with different architectures and hyperparameters to improve the quality of generatedimages.
• Consider techniques like progressive growing, spectral normalization, or feature matching tostabilize training and improve image quality.
8. Conclusion:
• Summarize the findings of the lab exercise, including the performance of the GAN model andany insights gained during experimentation.
• Discuss potential applications of GANs in image generation and future research directions forfurther exploration.
Program:
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
X_train = np.expand_dims(X_train, axis=-1)  # Ensure (28, 28, 1) shape
# Define Latent Space Dimension
latent_dim = 100
# Generator Model
def build_generator():
    inputs = Input(shape=(latent_dim,))
    x = Dense(128)(inputs)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(256)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(28 * 28 * 1, activation='tanh')(x)
    outputs = Reshape((28, 28, 1))(x)
    return Model(inputs, outputs)
# Discriminator Model
def build_discriminator():
    inputs = Input(shape=(28, 28, 1))
    x = Flatten()(inputs)
    x = Dense(512)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)
# Build and compile discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002, 0.5),
                      metrics=['accuracy'])
# Build Generator
generator = build_generator()
# Create Combined Model (Generator + Discriminator)
z = Input(shape=(latent_dim,))
generated_img = generator(z)
discriminator.trainable = False  # Freeze discriminator for combined model
validity = discriminator(generated_img)
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
# Training Parameters
epochs = 30000
batch_size = 128
# Ensure Reset of Loss Trackers Before Training
discriminator._loss_tracker = tf.keras.metrics.Mean(name="loss")
# Training Loop
for epoch in range(epochs):
    # Train Discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_imgs = generator.predict(noise)
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
    # Print Progress
    if epoch % 1000 == 0:
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

Output:
0 [D loss: 0.605581, acc.: 68.36%] [G loss: 0.917881]
1000 [D loss: 0.016042, acc.: 100.00%] [G loss: 4.820375]
2000 [D loss: 0.012413, acc.: 100.00%] [G loss: 5.390567]
...
29000 [D loss: 0.632694, acc.: 62.50%] [G loss: 1.293962]








Result:
Thus the Program to Image generation using GANhas been executed successfully
