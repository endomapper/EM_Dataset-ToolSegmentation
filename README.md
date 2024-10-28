# EM Dataset : Tool segmentation
**Authors** : Clara Tomasini, Iñigo Alonso, Luis Riazuelo, Ana C. Murillo.

This software performs tool segmentation on real endoscopy videos. 

### Related Publications:

Clara Tomasini, Iñigo Alonso, Luis Riazuelo and Ana C. Murillo, [**"Efficient tool segmentation for endoscopic videos in the wild"**](https://proceedings.mlr.press/v172/tomasini22a.html), *2022 International Conference on Medical Imaging with Deep Learning (MIDL)*, pp. 1218-1234 [PDF](https://proceedings.mlr.press/v172/tomasini22a/tomasini22a.pdf)

```
@inproceedings{tomasini2022efficient,
  title={Efficient tool segmentation for endoscopic videos in the wild},
  author={Tomasini, Clara and Alonso, I{\~n}igo and Riazuelo, Luis and Murillo, Ana C},
  booktitle={International Conference on Medical Imaging with Deep Learning},
  pages={1218--1234},
  year={2022},
  organization={PMLR}
}
```

This software has been trained and evaluated with a few sequences from the EndoMapper dataset, as described in:

Azagra P. et al. **Endomapper dataset of complete calibrated endoscopy procedures**. *Scientific Data*. 2023. Accepted for publication.

A video demo of the segmentations obtained on this dataset is available [here](https://drive.google.com/file/d/1anOHK4h19EesMFc_drYFnbcYtOBeTuJb/view?usp=sharing).

# 1. License
**EM Dataset: Tool segmentation** is released under AGPLv3 license. 

### Third-party code
This repository is built on a fork of projects [**robot-surgery-segmentation**](https://github.com/ternaus/robot-surgery-segmentation) (with MIT License), the official implementation of the paper 

[1] *Automatic Instrument Segmentation in Robot-Assisted Surgery using Deep Learning*. Shvets, Alexey A., et al. IEEE Int. Conf. on Machine Learning and Applications. 2018.

and [**MiniNet-v2**](https://github.com/Shathe/MiniNet-v2) (with AGPLv3 License), the official implementation of the paper

[2] *MiniNet: An Efficient Semantic Segmentation ConvNet for Real-time Robotic Applications*. Alonso, Iñigo et al. IEEE Transactions on Robotics. 2020.

# 2. Prerequisites
The software has been tested on **Ubuntu 20.04** and uses [Python](https://www.python.org). **Required 3.X**.

### Required packages:

* PyTorch 0.4.0
* TorchVision 0.2.1
* Tensorflow 
* Numpy 1.14.0
* Opencv-python 3.3.0.10
* Tqdm 4.19.4
* Albumentations
* Matplotlib

# 3. Proposed pipeline

Our proposed pipeline to segment tools in full endoscopy videos includes a classifier to determine whether or not to perform the segmentation (if a tool is in the frame or not), and MiniNet to perform the segmentation. 
![results](/images/pipeline_efficient.png)

# 4. How to run

Folder **endovis_challenge** contains files adapted from **robot-surgery-segmentation** for models LinkNet and UNet. 
Folder **mininet** contains files adapted from **MiniNet-v2** as well as the implementation of our clasifier. 

### Training

* File *endovis_challenge/train_ft.sh* performs training of LinkNet and UNet models. 
* File *mininet/train.sh* performs training of MiniNet model. 
* File *mininet/train_classif.sh* performs training of our clasifier. 

### Testing

Fine-tuned models and trained clasifier are available [here](https://drive.google.com/drive/folders/1BYyfUek6arVhpgChWuhD6JVQ9-RS4ZNm?usp=sharing). 

File *mininet/generate_masks.py* provides an example of how to use the full segmentation pipeline including MiniNet model and our clasifier in order to get a prediction for a given image.
Create folder *data/images* at main to store images on which to apply the segmentation.

# 5. Results 

Models UNet and LinkNet were available in **robot-surgery-segmentation** pretrained on images from the EndoVis17 dataset, and were then fine-tuned on more specific images from EndoMapper dataset. Mininet was trained from scratch on EndoVis17 dataset and then fine-tuned on EndoMapper dataset.

The following figure shows examples of binary segmentations from EndoMapper dataset obtained using different models without applying our pre-filtering clasifier. 
The last column shows an example of a frame without tool in which all segmentation models introduce False Positives. The effect of our pre-filtering classifier can be seen in the [Video Demo](https://drive.google.com/file/d/1anOHK4h19EesMFc_drYFnbcYtOBeTuJb/view?usp=sharing).


![results](/images/results_seg_hculb.png)
