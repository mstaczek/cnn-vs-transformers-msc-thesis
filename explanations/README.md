# Folder where the computed explanation pickles are stored

Each folder is specific for one dataset but contains many subfolders, one for each model evaluated. 

Each subfolder has a structure that can be used with [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) from the torch library except that instead of images there are **pickled explanations of each image**.