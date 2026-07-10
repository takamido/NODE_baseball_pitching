## NODE model for predicting pitching motion
This repository contains code and datasets for prediction and modeling of baseball pitching motion using a NODE (Neural Ordinary Differential Equation) model [1]. This NODE model predicts the remaining 92% of the pitching motion based on the initial 8% of the motion after its onset. For technical details, please refer to the original paper [1].

## Folders
- data: motion data of eight pitchers. Each dataset is stored in MAT format.
  
- program: programs for the NODE model. main_program.ipynb is the main script for training and evaluating the model. The other Python files are supplementary programs that contain the functions used by the project.　These programs are currently configured for (d = 2), which is the optimal number of latent dimensions. For details, please refer to the original paper [1].
  
- results: a folder containing the results. The figures in the paper can be reproduced using the files in this directory.
  
- trained_model: a folder containing the pretrained models. For 10-fold cross-validation, ten models are generated for each pitcher.
  
- videos: a folder containing demo videos and visualizations of the results.

## How to run
The implementation is based on Google Colab. Please download the code, data, and pretrained models, and set the paths in the main program to match your environment. You can then reproduce the model training and testing by running the main program.

## Reference
[1] coming soon
