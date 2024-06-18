# Results

Here I document all experiments.

### ToC:

- [20240410-gradcam-256](#20240410-gradcam-256)
- [20240410-gradcam-1024](#20240410-gradcam-1024)
- [20240410-kernelshap-64](#20240410-kernelshap-64)
- [20240414-1-2x2-pca-comparison](#20240414-1-2x2-pca-comparison)
- [20240414-2-gradcam-1024-fixed-efficientnet-b4](#20240414-2-gradcam-1024-fixed-efficientnet-b4)
- [20240414-3-2x2-pca-comparison-fixed-efficientnet-b4](#20240414-3-2x2-pca-comparison-fixed-efficientnet-b4)
- [20240417-gradcam-256-only-same-prediction](#20240417-gradcam-256-only-same-prediction)
- [20240421-fixing-kernelshap](#20240421-fixing-kernelshap)
- [20240505-finetuned-gradcam-256-ig-64-kernelshap-64](#20240505-finetuned-gradcam-256-ig-64-kernelshap-64)
- [20240519-print-gradcam-resolutions](#20240519-print-gradcam-resolutions)
- [20240521-compare-kernelshap-steps](#20240521-compare-kernelshap-steps)
- [20240531-accuracy-not-resized](#20240531-accuracy-not-resized)
- [20240603-gradcam-512-histograms-clustering](#20240603-gradcam-512-histograms-clustering)
- [20240613-sample-explanations-kernelshap-ig](#20240613-sample-explanations-kernelshap-ig)
- [20240616-adding-stdev-to-metric](#20240616-adding-stdev-to-metric)
- [20240618-visualizations](#20240618-visualizations)

## 20240410-gradcam-256

Settings:
- Models: ['DeiT_S', 'DeiT_T', 'DenseNet121', 'EfficientNet_B3', 'EfficientNet_B4', 'ConvNeXtV2_Nano', 'PyramidViT_V2_B2', 'MobileNetV3', 'Swin_T', 'ResNet18', 'ResNet50', 'ResNeXt50', 'Res2Net50', 'ViT_B_32']
- Method: GradCAM
- Images: 16 batches by 16 images, 256 total

Took 150s with T4 Colab GPU.

Results:

Cosine similarity + PCA | RBF similarity + PCA
:-------------------------:|:-------------------------:
![](20240410-gradcam-256/pca_cosine.png)  |  ![](20240410-gradcam-256/pca_rbf.png)

Clusters:
- DenseNet121, EfficientNet_B3, PyramidViT_V2_B2, MobileNetV3, Swin_T, ResNet18, ResNet50, ResNeXt50, Res2Net50,
- ConvNeXtV2_Nano, DeiT_S, DeiT_T, ViT_B_32,
- EfficientNet_B4 

Note: EfficientNet_B4 behaves strangely - almost all explanations highlight top right corner. Could be somewhat fixed by replacing it with an average. See below for a visualized example of this bug.

## 20240410-gradcam-1024

Settings:
- Models: ['DeiT_S', 'DeiT_T', 'DenseNet121', 'EfficientNet_B3', 'EfficientNet_B4', 'ConvNeXtV2_Nano', 'PyramidViT_V2_B2', 'MobileNetV3', 'Swin_T', 'ResNet18', 'ResNet50', 'ResNeXt50', 'Res2Net50', 'ViT_B_32']
- Method: GradCAM
- Images: 32 batches by 32 images, 1024 total

Took 520s with T4 Colab GPU.

Results:

Cosine similarity + PCA | RBF similarity + PCA
:-------------------------:|:-------------------------:
![](20240410-gradcam-1024/pca_cosine.png)  |  ![](20240410-gradcam-1024/pca_rbf.png)

Clusters - Same as 20240410-gradcam-256 (for 256 instead of 1024 analyzed images).

Note: EfficientNet_B4 behaves strangely - same as above, almost all explanations highlight top right corner. Could be somewhat fixed by replacing it with an average.

Good - some CNN (EfficientNet_B3) | Bad - EfficientNet_B4 (CNN)
:-------------------------:|:-------------------------:
![](20240410-gradcam-1024/good_EfficientNet_B3.png)  |  ![](20240410-gradcam-1024/bad_EfficientNet_B4.png)

## 20240410-kernelshap-64

Settings:
- Models: ['DeiT_S', 'DeiT_T', 'DenseNet121', 'EfficientNet_B3', 'EfficientNet_B4', 'ConvNeXtV2_Nano', 'PyramidViT_V2_B2', 'MobileNetV3', 'Swin_T', 'ResNet18', 'ResNet50', 'ResNeXt50', 'Res2Net50', 'ViT_B_32']
- Method: KernelSHAP
- Images: 4 batches by 16 images, 64 total

Took 1550s with T4 Colab GPU.

Results:

Cosine similarity + PCA | RBF similarity + PCA
:-------------------------:|:-------------------------:
![](20240410-kernelshap-64/pca_cosine.png)  |  ![](20240410-kernelshap-64/pca_rbf.png)

Clusters:
- ConvNeXtV2_Nano, DeiT_S, DeiT_T, EfficientNet_B4, PyramidViT_V2_B2, Swin_T, ResNet18, Res2Net50, ViT_B_32,
- EfficientNet_B3, MobileNetV3, ResNet50
- DenseNet121, ResNeXt50

Note: Explanations generally look bland. Possibly, changing KernelSHAP settings (64 segments and sampled 100 times) to increase how many times the superpixels are sampled, would improve the results. Obviously, segmentation is not perfect too and is an additional variable.

Good explanation | Bad explanation
:-------------------------:|:-------------------------:
![](20240410-kernelshap-64/good_DeiT_S.png)  |  ![](20240410-kernelshap-64/bad_ResNeXt50.png)

## 20240414-1-2x2-pca-comparison

Take PCA plots from `20240410-gradcam-1024` and `20240410-kernelshap-64` for cosine and RBF similarity and plot them side by side:

![](20240414-1-2x2-pca-comparison/pca_gradcam_kernelshap_comparison.png)

## 20240414-2-gradcam-1024-fixed-efficientnet-b4

Settings - rerun `20240410-gradcam-1024` for EfficientNet_B4 only after fix.

Fix: replace top-right corner with an average of other pixels.

Results: extreme value from top-right corner has been removed for all explanations -> 

Before fix | After fix
:-------------------------:|:-------------------------:
![](20240410-gradcam-1024/bad_EfficientNet_B4.png) | ![](20240414-2-gradcam-1024-fixed-efficientnet-b4/fixed_EfficientNet_B4.png)

## 20240414-3-2x2-pca-comparison-fixed-efficientnet-b4

Goal: repeat `20240414-1-2x2-pca-comparison` with fixed EfficientNet_B4.

Note: only GradCAM PCA plots were recomputed.

![](20240414-3-2x2-pca-comparison-fixed-efficientnet-b4/pca_gradcam_kernelshap_comparison.png)

## 20240417-gradcam-256-only-same-prediction

Goal: 

- count the number of images on which a given pair of models gave the same prediction (not necessarily correct, but the same),
- compute similarity matrices for cosine and rbf but using only images for which a given pair of models gave the same prediction,
- compute similarity between this matrix of counts of same predictions and all similarity matrices from earlier.

Setting:

- as in `20240410-gradcam-256`, but for 8 batches of 32 images, same 256 total.

Took 300s with T4 GPU on Colab.

### Results:

Correlations between a matrix of count of same predictions and different similarity matrices (rbf/cosine and using [all/only matching] predictions):

|               | All images | Only matching predictions |
|---------------|------------|---------------------------|
| Cosine        | 0.695880   | 0.672233                  |
| RBF           | 0.642113   | 0.641272                  |

Counts heatmap: 

![](20240417-gradcam-256-only-same-prediction/same_predictions_fractions_heatmap.png)

Similarity matrices that it was compared to:

|               | All images | Only images with matching prediction |
|---------------|--------|-----|
| Cosine        | ![](20240417-gradcam-256-only-same-prediction/heatmap_cosine_all.png) | ![](20240417-gradcam-256-only-same-prediction/heatmap_cosine_only_matching.png) |
| RBF           | ![](20240417-gradcam-256-only-same-prediction/heatmap_rbf_all.png) | ![](20240417-gradcam-256-only-same-prediction/heatmap_rbf_only_matching.png) |

PCA obtained for these similarity matrices:

|               | All images | Only images with matching prediction |
|---------------|--------|-----|
| Cosine        | ![](20240417-gradcam-256-only-same-prediction/pca_cosine_all.png) | ![](20240417-gradcam-256-only-same-prediction/pca_cosine_only_matching_prediction.png) |
| RBF           | ![](20240417-gradcam-256-only-same-prediction/pca_rbf.png) | ![](20240417-gradcam-256-only-same-prediction/pca_rbf_only_matching_prediction.png) |

## 20240421-fixing-kernelshap

Goal: Change KernelSHAP settings to improve the quality of explanations.

Settings:

Before: 64 segments and sampled 100 times. New: 50 segments and sampled 100 times. Reason: In my Bsc thesis we based on KernelSHAP with 50 segments and sampled 50 times by default, so possibly here it should also work. Also, changed the way explanations are computed: one image at a time, not in batches.

Results: Not much improvement. Possibly, further increasing the number of samples would help. Also, segmentation is not perfect and has some influence.

|               | Before | After |
|---------------|--------|-------|
| ConvNeXtV2_Nano - little change | ![](20240421-fixing-kernelshap/convnextv2_before.png) | ![](20240421-fixing-kernelshap/convnextv2_after.png) |
| ConvNeXtV2_Nano - slight change | ![](20240421-fixing-kernelshap/convnextv2_before_2.png) | ![](20240421-fixing-kernelshap/convnextv2_after_2.png) |
| ResNet18      | ![](20240421-fixing-kernelshap/resnet_before.png) | ![](20240421-fixing-kernelshap/resnet_after.png) |
| ResNet18      | ![](20240421-fixing-kernelshap/resnet_before_2.png) | ![](20240421-fixing-kernelshap/resnet_after_2.png) |

## 20240505-finetuned-gradcam-256-ig-64-kernelshap-64

Goal: Add new explanation method (Integrated Gradients) and finetune models on Imagenette2 and see if the results are better. 

Settings: 
- use `timm` and finetune each models with `vision_learner.fine_tune` for 1 epoch. 
- compute explanations for all of the models and 3 explanation methods. If no fintuned model was available, the original model was used (pretrained on Imagenet).
- Computed explanations for 256 images with GradCAM and for 64 for KernelSHAP and Integrated Gradients.
- Integrated Gradients was set with a default hyperparameter value of 50 steps.

### Finetuning results:

- funetuning took up to 5 minutes per model on Colab with T4 GPU,
- accuracies of models are around 97%-99% on Imagenette2,
- some models failed to finetune due to timm and fastai errors (MobileNetV3 and Swin_T),
- explanations for GradCAM are almost the same. For KernelSHAP they look different but neither better nor worse.

| Model | Before | After |
|---|---|---|
| ResNet18 | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/old-gradcam-resnet18-1.png) | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/new-gradcam-resnet18-1.png) |
| ConvNeXtV2_Nano | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/old-kernelshap-convnext-1.png) | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/new-kernelshap-convnext-1.png) |

### Integrated Gradients

Sample explanations:
| ResNet18 | EfficientNet_B3 | ViT_B_32 |
|---|---|---|
| ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/ig-resnet18.png) | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/ig-efficientnet-b3.png) | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/ig-vit-b-32.png) |

Comment: Integrated Gradients look best without background image. In general, they are single-pixel dots, here visible in the first 2 explanations as small artefacts on the dog head or body.

### Similarities between models

For cosine similarity:

| Explanation | All images | Only matching predictions |
|---|---|---|
| KernelSHAP | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_kernelshap_cosine_all.png) | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_kernelshap_cosine_only_matching.png) |
| Integrated Gradients | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_integratedgradients_cosine_all.png) | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_integratedgradients_cosine_only_matching.png) |
| GradCAM | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_gradcam_cosine_all.png) | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_gradcam_cosine_only_matching.png) |

For RBF similarity:

| Explanation | All images | Only matching predictions |
|---|---|---|
| KernelSHAP | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_kernelshap_rbf_all.png) | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_kernelshap_rbf_only_matching.png) |
| Integrated Gradients | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_integratedgradients_rbf_all.png) | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_integratedgradients_rbf_only_matching.png) |
| GradCAM | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_gradcam_rbf_all.png) | ![](20240505-finetuned-gradcam-256-ig-64-kernelshap-64/pca_gradcam_rbf_only_matching.png) |


### Comments

- Similarity of models after applying GradCAM explanation method seem to give similar PCA plots for both cosine and RBF similarity metrics.
- There is no clear similarity between PCA plots of models similarity generated with different explanation methods or metrics.


## 20240519-print-gradcam-resolutions

Goal: Get resolution of GradCAM explanations for all models.

Results:

| Model Name | Resolution |
|---|---|
| ConvNeXt V2-N | 7x7 |
| DeiT-S | 14x14 |
| DeiT-T | 14x14 |
| DenseNet121 | 7x7 |
| EfficientNet-B3 | 7x7 |
| EfficientNet-B4 | 7x7 |
| MobileNetV3 | 7x7 |
| PVTv2-B2 | 7x7 |
| ResNet18 | 7x7 |
| ResNet50 | 7x7 |
| ResNeXt50 | 7x7 |
| Res2Net50 | 7x7 |
| Swin-T | 7x7 |
| ViT-B/32 | 7x7 |


## 20240521-compare-kernelshap-steps

Goal: Compare KernelSHAP explanations for different number of steps.

Settings: image segmented into 50 superpixels. Number of steps: 50, 100, 150, 200, 250, 300, 400. Explained image is of class fish, model is ResNet18 and the image is:

![](20240521-kernelshap-more-samples/n01440764_1185.JPEG)

Results:

Time-wise, there is a linear increase in time with the number of steps.

![](20240521-kernelshap-more-samples/output.png)


## 20240531-accuracy-not-resized

Goal: Check if using same transforms in dataloader affects models accuracy.

Setting: 
- Predict 1024 images using model-specific transforms (for all models).
- Predict 1024 images using same transforms for all models - Resize straight to 224x224 and Normalize.
- Calculate the difference in accuracy.

Results:

| Imagenette2 split: train | Imagenette2 split: val |
|---|---|
| ![](20240531-accuracy-not-resized/diff_train.png) | ![](20240531-accuracy-not-resized/diff_val.png) |

Explanation: Models performance drops just by a few percent. For most models, the difference is that instead of getting center-cropped 224x224 pixels, they get full 256x256 images. EfficientNet models expect slightly larger images (B3 - 288, B4 - 320) and have visibly larger drop in accuracy. However, ViT model was the only with different Normalization of the images:
- all models: `Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))`
- ViT: `Normalize(mean=tensor([0.5, 0.5, 0.5]), std=tensor([0.5, 0.5, 0.5]))`

Models accuracies can be found in the `20240531-accuracy-not-resized` folder.

Further investigation:
- Take ViT model, take same common transformation as earlier (Resize to 224x224 and Normalize) and change Normalization to the model-specific one.

Results:

Accuracy difference between ViT model with model-specific transforms vs common transforms with model-specific Normalization:

| Model | Imagennete2 split: train | Imagenette2 split: val |
|---|---|---|
| ViT_B_32 | 0.03 | 0.03 |

Conclusion: The same transformations cannot be used for all models. However, CenterCrop can be ommited, and resizing images to the expected 224x224 size is ok. Normalization should be model-specific though, in particular, ViT model has to use it's normalization.

## 20240603-gradcam-512-histograms-clustering

### Part 1 - rerun GradCAM + histograms

Goal: rerun GradCAM after last fixes, create histograms of similarities between every pair of explanations.

Settings: Gradcam for 512 images, all models, cosine and RBF similarity.

Results:

| Similarity metric | Histograms |
|---|---|
| Cosine | ![](20240603-gradcam-512-histograms-clustering/histograms_gradcam_cosine.png) |
| RBF | ![](20240603-gradcam-512-histograms-clustering/histograms_gradcam_rbf.png) |

Histograms confirm that some models are more different than the other, with no clear distinction between CNN and transformer models.

### Part 2 - clustering

Goal: for each csv (cosine and RBF similarity) create a hierachical clustering dendrogram, PCA plot as usually and graph community detection results.

Settings: 

- Analysis i sbased on the Part 1 of the experiment. 
- Hierachical clustering colors are manually selected. 
- Graph community detection is based on the Louvain algorithm.

Results:

| Visualization method | Cosine | RBF|
|---|---|---|
| PCA | ![](20240603-gradcam-512-histograms-clustering/pca_gradcam_cosine_all.png) | ![](20240603-gradcam-512-histograms-clustering/pca_gradcam_rbf_all.png) |
| Hierachical clustering | ![](20240603-gradcam-512-histograms-clustering/dendrogram_hierarchical_gradcam_cosine_all.png) |  ![](20240603-gradcam-512-histograms-clustering/dendrogram_hierarchical_gradcam_rbf_all.png) | 
| Graph community detection | ![](20240603-gradcam-512-histograms-clustering/graph_louvain_gradcam_cosine_all.png) | ![](20240603-gradcam-512-histograms-clustering/graph_louvain_gradcam_rbf_all.png) |


Conclusions: Results are similar to each other, with no clear distinction between CNN and transformer models. Visualization methods are consistent with each other. PCA is the least readable and dendrogram is the most readable. Differences in values of the metrics influence the values or shapes in the plots but not the meaning.

## 20240613-sample-explanations-kernelshap-ig

Goal: compute a few sample explanations for KernelSHAP and Integrated Gradients. Vizualize them side by side.

Settings:

- KernelSHAP: 50 segments, 350 samples,
- Integrated Gradients: 100 steps,
- include only images for which all models gave the same prediction.

Results:

| Sample image 1 | Sample image 2 | Sample image 3 | Sample image 4 |
|---|---|---|--|
| ![](20240613-sample-explanations-kernelshap-ig/explanations_n02102040_2644.png) | ![](20240613-sample-explanations-kernelshap-ig/explanations_n03028079_49234.png) | ![](20240613-sample-explanations-kernelshap-ig/explanations_n03417042_11534.png) | ![](20240613-sample-explanations-kernelshap-ig/explanations_n03000684_15194.png) |

Conclusions: 

- Quality of explanations vary between images. 
- KernelSHAP with same mask can make little sense if a tiny part of the image matters (eg. dog).
- Integrated Gradients sometimes are random for ViT/DeiT.
- Visible differences in GradCAM explanations between models.

## 20240616-adding-stdev-to-metric

Goal: Instead of relying solely on mean of similarity metric, incorporate also standard deviation.

Settings:
- to this point: a mean of similarity metric was returned
- new: return $\sqrt{(1-mean)^2 + stdev^2}$, which is a distance from (0,0) of a point with:
  - X = 1 - mean similarity = dissimilarity
  - Y = stdev of similarity

Results:

| Explanation | Histograms of values from models' similarity matrix|
|---|---|
| KernelSHAP | ![](20240616-adding-stdev-to-metric/histograms_KernelSHAP_8.png) |
| Integrated Gradients | ![](20240616-adding-stdev-to-metric/histograms_IntegratedGradients_8.png) |
| GradCAM | ![](20240616-adding-stdev-to-metric/histograms_GradCAM_512.png) |

Conclusion: Values of metric are similar to each other for both cosine similarity and RBF similarity. Value 80 of scaling parameter for RBF works well for this case.

## 20240618-visualizations


### Part 1 - rerun GradCAM histograms

Goal: rerun creating histograms of similarities between every pair of GradCAM explanations.

Settings: Gradcam for 512 images, all models, cosine and RBF similarity after fixing computing similarities of a pair of images.

Results:

| Similarity metric | Histograms |
|---|---|
| Cosine | ![](20240618-visualizations/histograms_gradcam_cosine.png) |
| RBF | ![](20240618-visualizations/histograms_gradcam_rbf.png) |

No visible difference compared to the earlier version.

### Part 2 - clustering

Goal: for each csv (cosine and RBF similarity) create a hierachical clustering dendrogram, and graph community detection results.

Settings: 

- Analysis i sbased on the Part 1 of the experiment. 
- Hierachical clustering has disabled colors in dendrograms. 
- Graph community detection is based on the Louvain algorithm.

Results:

| Visualization method | Cosine | RBF|
|---|---|---|
| PCA | not used anymore | not used anymore |
| Hierachical clustering | ![](20240618-visualizations/dendrogram_hierarchical_gradcam_cosine_all.png) |  ![](20240618-visualizations/dendrogram_hierarchical_gradcam_rbf_all.png) | 
| Graph community detection | ![](20240618-visualizations/graph_louvain_gradcam_cosine_all.png) | ![](20240618-visualizations/graph_louvain_gradcam_rbf_all.png) |