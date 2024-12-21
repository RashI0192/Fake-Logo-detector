<h1 style="color: #3498db; text-align: center;">Fake Logo Detector</h1>
<br/>
This is imported from my kaggle - https://www.kaggle.com/code/ray0911/fake-logo-detection-using-tensorflow-keras.
<br/>
Version 4 - Data cleaning and EDA done
<br/>
version 6- finishes the CNN network 
<br/>
Version 7 - Complete final code - included here. 
<br/>

Below Is a Brief Explaination
<br/>
Modules used - Numpy,Pandas,matplotlib.pyplot ,Tensorflow,Keras
<br/>
<h1 style="color: #3498db; text-align: center;">Work Flow</h1>

<ol>
   <li>Data Cleaning And EDA
      <ul>
         <li> Uneccesary comoulm - Tagline Column Dropped as only working with the images</li>
         <li> Ensuring No Null/Duplicate values</li>
         <li>Converted Categorical values ['geniune ','fake'] into numerical for compatibility</li>
         <li> Checked for bias in data by comparing the unique labled values.</li>
         <li>Shuffled Dataset to elimate any ordering bias that may impact model training.</li>
         <li>For my own convience I adjusted the filepaths. The orignal filepaths were messy. I simplified access to image files stored in multiple folders by standardizing
   their paths. This helped me with  image loading in subsequent steps.</li>
   <li>Image Display</li>
         
         
      </ul>
   </li>
   <li>Dataset Spliting
   </li>
</ol>
<br/>
1. Data Cleaning And EDA :
   - a) Uneccesary comoulm - Tagline Column Dropped as only working with the images
   - b)Ensuring No Null/Duplicate values
   - c)Converted Categorical values ['geniune ','fake'] into numerical for compatibility
   - d) Checked for bias in data by comparing the unique labled values.
   - e) Shuffled Dataset to elimate any ordering bias that may impact model training.
   - f) For my own convience I adjusted the filepaths. The orignal filepaths were messy. I simplified access to image files stored in multiple folders by standardizing
   their paths. This helped me with  image loading in subsequent steps.
   - g) Image Display

2. Dataset Spliting
   a) I used get_dataset_partitions_tf function to split my dataset into  training, validation and test . The function created ensured reproducibilty by setting a seed for sufflting , Shufftling the dataset prevents the model from learning unintentional pattern.
   b) each subset is created with the correct size ensuring a balnced split for reliable training.
   c) 80/10/10 Split - the majority of the data is used for training to ensure the model learns general patterns and features. Improve its ability to generalise
   10% Validation used 
A larger training set helps the model improve its ability to generalize.
10% Validation Set to tune hyperparameters and assess the model's performance during training. This is to prevent overfitting and show how well the model is generalising to unseen data during training.
10% Test split = final evaluation to check model's performance

  d) I also catched data to speed up the training process[CNN model I used at my first try was estimated to run for 5+hours due to unavaibility of GPU] . This ensured data is loaded into memory only once so less overhead for epochs.R
Also Prefetch data . This ensures that the  next batch of data is prepared while the current batch is being processed

3. Pre-Processing and Data Augmentation
   a)Resizingto standardizes all images to the same size (224x224) for consistent input to the model.
     Rescaling to normalise pixel values (scaling them to [0, 1]) =>model converges faster during training.
   b) Data augmentation does random flipping and rotation to introduce variations to the dataset. Hence, helping the model generalise better to unseen data by stimulating real world variations.


4. Model Architechture :
   a)Convolutional layers :
6 Conv2D Layers to detect local patterns like  edges ,corners,textures in image.
   ReLU activation function used. 
6 MaxPooling2D Layers to reduce the spatial dimensions of the feature maps [ pooling size - (2,2)to downsample feature maps]
1 Flatten Layer to convert the 2D feature maps from the convolutional layers into a 1D vector for input into fully connected layers.
2 Dense Layers:
First Dense Layer: 64 units with ReLU activation / higher-level representations.
Final Dense Layer: 2 units with Softmax activation for binary classification (Fake or Genuine).
Number of Dense Layers: 2

   b) Adam Optimiser to adapt learning rates during training for efficient convergence.
Loss function (SparseCategoricalCrossentropy) used for  binary classification tasks. 
accuracy metric to track performance during validation

 c) I trained for 52 epochs. This allowed the model to repeatedly learn and refine patterns.The validation data ensured t to detect overfitting early during training.
   

5. Training history and confusion matrix also used to evaluate the model performance

 



   
