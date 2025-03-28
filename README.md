## CPDet3D

<p align="center"><img src="./images/framework.png" alt="drawing" width="90%"/></p>

This project provides the code and results for 'Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection', CVPR 2025.

Anchors: Yun Zhu, [Le Hui](https://scholar.google.com/citations?user=se31JGQAAAAJ&hl=zh-CN), Hang Yang, [Jianjun Qian](https://scholar.google.com/citations?user=oLLDUM0AAAAJ&hl=zh-CN), [Jin Xie*](https://scholar.google.com/citations?user=Q7QqJPEAAAAJ&hl=zh-CN), [Jian Yang](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN)

PaperLink: https://arxiv.org/pdf/2503.21099


### Introduction
> Both indoor and outdoor scene perceptions are essential for embodied intelligence. However, current sparse supervised 3D object detection methods focus solely on outdoor scenes without considering indoor settings. To this end, we propose a unified sparse supervised 3D object detection method for both indoor and outdoor scenes through learning class prototypes to effectively utilize unlabeled objects.
Specifically, we first propose a prototype-based object mining module that converts the  unlabeled object mining into a matching problem between class prototypes and unlabeled features. By using optimal transport matching results, we assign prototype labels to high-confidence features, thereby achieving the mining of unlabeled objects. We then present a multi-label cooperative refinement module to effectively recover missed detections through pseudo label quality control and prototype label cooperation. Experiments show that our method achieves state-of-the-art performance under the one object per scene sparse supervised setting across indoor and outdoor datasets. With only one labeled object per scene, our method achieves about 78\%,  90\%, and  96\% performance compared to the fully supervised detector on ScanNet V2, SUN RGB-D, and KITTI, respectively, highlighting the scalability of our method.

### Environment settings

- To install the environment, we follow [SPGroup3D](https://github.com/zyrant/SPGroup3D).

- All the `CPDet3D`-related code locates in the folder [projects_sparse](projects_sparse/configs).


### Data Preparation

- Follow the mmdetection3d data preparation protocol described in [scannet](data/scannet/README.md), [sunrgbd](data/sunrgbd/README.md). 


### Label Preparation
- We provide the indoor sparse supervised split of `scannet_infos_train.pkl` and `sunrgbd_infos_train.pkl` on [GoogleDrive](https://drive.google.com/drive/folders/1mr9BA7wxBvkiTk5uSpVPSe0gPB8My1CX?usp=sharing).
- Otherwise, you can generate your own `pkl` by [create_sparse_infos](tools/create_sparse_infos.py).


### Training

To start training, run [train](tools/train.py) with CPDet3D [configs](projects_sparse/configs), which includes two stages.


### Testing

Test pre-trained model using [test](tools/dist_test.sh) with CPDet3D [configs](projects_sparse/configs).

### Main Results
| Dataset | mAP@0.25 | mAP@0.5 | Download | config |
|:-------:|:--------:|:-------:|:--------:|:--------:|
| ScanNet V2 | 56.1 | 40.8 | [GoogleDrive](https://drive.google.com/drive/folders/1MPW8A0UsLTwGwE5dLwMPC5h5fhIu29kO?usp=sharing)  | [config](projects_sparse/configs/two_stage/tr3d_scannet-3d-18class_cpdet3d_2.py) |
| SUN RGB-D | 60.2 | 43.3| [GoogleDrive](https://drive.google.com/drive/folders/1MPW8A0UsLTwGwE5dLwMPC5h5fhIu29kO?usp=sharing) |[config](projects_sparse/configs/two_stage/tr3d_sunrgbd-3d-10class_cpdet3d_2.py) |


Due to the size of these datasets and the randomness that inevitably exists in the model,  the results on these datasets fluctuate significantly. It's normal for results to fluctuate within a range.

### Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{cpdet3d,
  title={Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection},
  author={Yun Zhu, Le Hui, Hang Yang, Jianjun Qian, Jin Xie, Jian Yang},
  booktitle={CVPR},
  year={2025}
}
```

### Acknowledgments

This project is based on the following codebases.
- [mmdetection3D](https://github.com/open-mmlab/mmdetection3d)
- [TR3D](https://github.com/SamsungLabs/tr3d)
- [ProtoSeg](https://github.com/tfzhou/ProtoSeg)

If you find this project helpful, please also cite the codebases above. Thanks.
