# Auto Chloro - A Plant Disease Classifier & Remedies Provider in Bangla
## About Auto Chloro
Auto Chloro is a plant disease classifier & remedies provider that uses deep learning. It can predict diseases and provide the remedies. The GUI is based on Bangla Language keeping in mind that, our primary target is to create an application to predict plant diseases and provide remedies for the Bangladeshi people.


## How to Use 

![select menu](images/gui1.PNG)


To predict the disease click on the "ছবি সিলেক্ট করুন" button.


![file open](images/gui2.PNG)


Select the image from your PC.


![confirmation](images/gui3.PNG)


Click on the selected image to predict the disease and get the remedies.


![remedies](images/gui4.PNG)


Finally, you get the disease name and remedies. 

## Details
### Dataset:
The dataset we will be using contains 17476 images. The train dataset has 16222 images belonging to 15 classes and the test dataset has 1254 images belonging to 15 classes.
Dataset link: https://www.kaggle.com/vasanthkumar14/plant-disease 
### Libraries:
1. Numpy 
2. Matplotlib
3. OS
4. Tensorlow
5. EasyGUI

### CNN Model:

We used a sequential model. The Sequential Model API is a way to build deep learning models that create a sequential class and create and add model layers to it. We used 4 convolutional layers with “Relu” (Rectified Linear Unit) activation functions. The parameters of the first conv2D are, filter-size, kernel-size, Input-shape. The convolutional layer is then pass to MaxPooling layer,pooliing size is the window size. We use Flatten to convert data into 1 Dimensional form. Dense layer feeds all outputs from the previous layer to all its neurons, each neuron providing one output to the next layer. Dropout function is a simple way to prevent overfitting. Dropout is a technique where randomly selected neurons are ignored during training. We used ‘Adam’ as our optimizer to optimize our data with learning rate. A metric is a function that is used to measure the model's performance. Here we take 15 epochs for train our model. More epochs increase the accuracy and decrease the loss (but it takes more time too).

### Saving the Model (h5):

It takes a lot of time to train the model. Therefore, we save the trained model so that we can save the time. Moreover, we have to use the saved model in our GUI, as it’s an application.

### GUI:

We used easygui, a simple GUI framework based on tkinter. At first, we load the saved model that we trained previously. Basically, we use the fileopenbox function to get the image path. Then, we load the image with load_img method. After that, we covert the image to array and expand the dimension where axis=0, it defines the index at which dimension should be inserted. If input has D dimensions then axis must have value in range [-(D+1), D].
We make a list of our labels and use multiple if-else statement to match our prediction with the diseases. Finally, we show the disease name and remedies in textbox.

### Current Status & Bugs

Currently we can show the disease properly if the image is good. Model's accuracy is 94-96%. The dataset is not good enough to predict the diseases properly every time.So, it does show wrong outputs sometimes. Here, we have 3 types of plants. The GUI is pretty simple and it was intentional, although it needs more work. Overall, the CNN model is good enough to show some good results and the whole code is working properly.

### Future Plans:

•	Increasing the data, based on Bangladeshi Crops/Plants
•	Creating a website and an app so that it can reach the people easily
•	Better design for the GUI
•	Adding this project to another one which is an automated cultivation system using IOT and ML.  

