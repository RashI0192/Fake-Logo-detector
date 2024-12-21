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


# Work Flow

## 1. Data Cleaning And EDA
- a) Unnecessary column - **Tagline Column** dropped as only working with the images.
- b) Ensured no null/duplicate values.
- c) Converted categorical values ['genuine', 'fake'] into numerical values for compatibility.
- d) Checked for bias in data by comparing the unique labeled values.
- e) Shuffled dataset to eliminate any ordering bias that may impact model training.
- f) For my own convenience, I adjusted the filepaths. The original filepaths were messy. I simplified access to image files stored in multiple folders by standardizing their paths. This helped with image loading in subsequent steps.
- g) Image display.

---

## 2. Dataset Splitting
- a) I used the `get_dataset_partitions_tf` function to split my dataset into training, validation, and test. The function ensured reproducibility by setting a seed for shuffling. Shuffling the dataset prevents the model from learning unintentional patterns.
- b) Each subset was created with the correct size, ensuring a balanced split for reliable training.
- c) **80/10/10 Split**:
  - The majority of the data (80%) is used for training to ensure the model learns general patterns and features, improving its ability to generalize.
  - 10% validation set is used to tune hyperparameters and assess the model's performance during training. This prevents overfitting and shows how well the model generalizes to unseen data during training.
  - 10% test split for final evaluation to check the model's performance.
- d) I also cached data to speed up the training process. (The CNN model I used at my first try was estimated to run for 5+ hours due to the unavailability of GPU.) This ensured data is loaded into memory only once, reducing overhead for epochs.
  - Prefetched data to ensure that the next batch of data is prepared while the current batch is being processed.

---

## 3. Pre-Processing and Data Augmentation
- a) **Resizing** to standardize all images to the same size (224x224) for consistent input to the model.
  - **Rescaling** to normalize pixel values (scaling them to [0, 1]) so the model converges faster during training.
- b) Data augmentation does random flipping and rotation to introduce variations to the dataset. This helps the model generalize better to unseen data by simulating real-world variations.

---

## 4. Model Architecture
- a) **Convolutional Layers**:
  - 6 Conv2D Layers to detect local patterns like edges, corners, and textures in images.
  - **ReLU activation function** used.
  - 6 MaxPooling2D Layers to reduce the spatial dimensions of the feature maps. (Pooling size - (2,2) to downsample feature maps.)
  - 1 Flatten Layer to convert the 2D feature maps from the convolutional layers into a 1D vector for input into fully connected layers.
  - 2 Dense Layers:
    - **First Dense Layer**: 64 units with ReLU activation for higher-level representations.
    - **Final Dense Layer**: 2 units with Softmax activation for binary classification (Fake or Genuine).
  - **Number of Dense Layers**: 2.
- b) **Adam Optimizer** to adapt learning rates during training for efficient convergence.
  - **Loss Function**: SparseCategoricalCrossentropy used for binary classification tasks.
  - **Accuracy Metric**: Used to track performance during validation.
- c) Trained for **52 epochs**. This allowed the model to repeatedly learn and refine patterns. The validation data ensured overfitting detection during training.

---

## 5. Evaluation
- **Training History**: Used to evaluate model performance across epochs.
- **Confusion Matrix**: Analyzed to evaluate the model's ability to distinguish between Fake and Genuine logos.


 



   
