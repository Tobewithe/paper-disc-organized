# Visualizing the Cure: From Distribution to Tensors

## 1. The Dataset Challenge: Instance Crowding Index (ICI)
First, we visualized the mathematical distribution of your proposed $ICI$ metric across the 1752 test instances. As shown below, **nearly 50% of the dataset** falls into the "Heavy Crowding" or "Extreme Adhesion" bins ($ICI > 0.5$). 

![ICI Distribution](./images/ici_distribution.png)

## 2. Breaking the Facade: Raw Mask Visualization
To physically prove our theory, we found the most severely overlapping pair of ground truth pigs in the test set (Image 144, Bounding Box IoU = 0.72). 

We extracted the **Raw Uncropped Masks** (the direct output of the matrix multiplication between Coefficients and Prototypes, *before* bounding box cropping) for both the Baseline model and the CCL model.

![Raw Mask Comparison](./images/raw_mask_comparison.png)

**Observation:**
* **Top Row (Baseline)**: Notice how Pig A's mask and Pig B's mask both heavily illuminate the shared overlapping region. The network struggles to differentiate them natively, relying entirely on the rectangular bounding box (not shown here) to forcefully cut the masks apart.
* **Bottom Row (CCL)**: Notice the absolute separation! Pig A's mask completely ignores the spatial area belonging to Pig B, and vice versa. By forcing the coefficients to be orthogonal during training, the network has learned to generate natively independent masks without needing a bounding box to "clean up" the bleeding.

