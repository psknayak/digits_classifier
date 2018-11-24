#%% [markdown]
# # Task 1A: Build a Fully Connected 2 - Layer Neural Network to Classify Digits
# 
# This notebook will serve as implementation of the API that you have created in your "Code" folder. You will write functions in the "py" files and use them here.
# 
# We will be using inbuilt MNIST dataset present in PyTorch and train a neural network to classify digits. 
#%% [markdown]
# ## What is expected from this notebook?
# 
# This notebook should be used to present your work. You should explain wherever necessary (but also not too much) about what you did and why you did it. You should explain things like hyper parameter settings (even if it was provided before hand to you by us), training performance and testing performance of the model. You should reason why your model is working fine and not overfitting.
# 
# Since numbers don't are an argot, you should also use visualizations wherever possible. You can visualize things like loss curve, show confusion matrix, etc. 
# 
# Finally, you can show some manual verifications by displaying and making predictions on random test examples. 
# 
# **NOTE: The amount of things you can do in this notebook is limitless (hyperbole). But don't do too much at the cost of rest of your tasks. Remember to maintain the outputs while submitting this notebook.**
#%% [markdown]
# ## Absolutely required items?
# 
# 1. First of all, import the libraries and the dataset.
# 2. Next, show dataset samples and distribution of different type of data. For example, in case of MNIST you can show some random images and their labels. Also, show distribution of each class of images.
# 3. Next, perform required transformations on MNIST dataset (normalization, scaling, grayscaling if required, etc) using torchvision transforms.
# 4. Create required dataloaders on PyTorch MNIST dataset to load data in mini-batches.
# 5. Train the model, show loss and accuracy at each step of operation.
# 6. Plot the **loss curve for both train and validation phase**
# 7. Pick some manual random images from test dataset and predict their values **showing expected and actual result**.
# 
# **NOTE: You may or may not delete these instruction cells after completion of the notebook.**
#%% [markdown]
# # Your solution

#%%
# Start coding from here


