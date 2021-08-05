# Diagnosis-of-Pneumonia-using-Chest-X-Ray-Images
This demonstrates the task of classifying Chest X-Ray images as either Normal or having Pneumonia. This task is avidly famous on the Kaggle platform as Chest X-Ray Images (Pneumonia). I propose 2 Deep Learning Architectural Neural Network Models, one; is a CNN Model, which I call the X-Ray CNN; which is tuned and trained from scratch, while the other is the famous ResNet50 Transfer Learning Model.

# Note: 

It contains the 2 (.pynb) files, 2 (.py) files, 4 model checkpoints (.pt), 1 requirements.txt file, 3 folders - figs, logs, model_checkpoints, 1 report paper (.pdf),.

First, open the DL_Prashant20200126-README.ipynb file.
The DL_Prashant20200126-README.ipynb file contains the whole implementation of the project run this first so that the saved model checkpoints will be created in the "model_checkpoints" folder (.pt files); move these .pt files to the main folder one path outside; further open the Testing-Submitted-Models_Prashant20200126.ipynb notebook, which is used to give proof to the results explained in my Project Report Paper & load the model checkpoints, which loads the metrics similar to what we see in my Report Paper.
There is nothing to be set or updated for the user to run my project, just place the dataset folder as suggested and let the other files be there as it is, and run the whole notebook to see the project implementation and results. The 2 submitted notebooks act as README for my Project, as I have made full use of the markdown comments, just like we see in Kaggle.

# Additional requirements:

    Please create a folder named "model_checkpoints" exactly in the main "Chest_X_Ray" folder, this is very important for saving the model_checkpoints
