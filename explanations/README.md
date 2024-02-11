# Folder where the computed explanation pickles are stored

Each folder is specific for one dataset and contains explanations for every evaluated model and explanation method, example:

```
explanations
└───imagenette2
    └───gradcam
        └───resnet18
            (train -> 10 classes -> .npy files)
        └───resnet50
            (train -> 10 classes -> .npy files)
```

Each subfolder has a structure that can be used with [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) from the torch library except that instead of images there are **pickled explanations of each image**.