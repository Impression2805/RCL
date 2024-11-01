# RCL
PyTorch implementation of our CVPR2024 paper "RCL: Reliable Continual Learning for Unified Failure Detection"

### [CVPR 2024] RCL: Reliable Continual Learning for Unified Failure Detectio
Fei Zhu, Zhen Cheng, Xu-Yao Zhang, Cheng-Lin Liu,  Zhaoxiang Zhang<br>

[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_RCL_Reliable_Continual_Learning_for_Unified_Failure_Detection_CVPR_2024_paper.pdf)

### Abstract
Deep neural networks are known to be overconfident for what they don’t know in the wild, which is undesirable for decision-making in high-stakes applications. Despite quantities of existing works, most of them focus on detecting out-of-distribution (OOD) samples from unseen classes, while ignoring large parts of relevant failure sources like misclassified samples from known classes. In particular, recent studies reveal that prevalent OOD detection methods are actually harmful for misclassification detection (MisD), indicating that there seems to be a tradeoff between those two tasks. In this paper, we study the critical yet under-explored problem of unified failure detection, which aims to detect both misclassified and OOD examples. Concretely, we identify the failure of simply integrating learning objectives of misclassification and OOD detection, and show the potential of sequence learning. Inspired by this, we propose a reliable continual learning paradigm, whose spirit is to equip the model with MisD ability first, and then improve the OOD detection ability without degrading the already adequate MisD performance. Extensive experiments demonstrate that our method achieves strong unified failure detection performance.

### Usage 
We run the code with torch version: 1.10.0, python version: 3.9.7
* Run our method given pretrained model CRL [ICML20] or FMFP [ECCV22 & TPAMI23]
```
python main_cvpr24.py
```
* The pretrained model can be get by runing the following codebase
* <https://github.com/Impression2805/FMFP>

### Reference
Our implementation references the codes in the following repositories:
* <https://github.com/Impression2805/OpenMix>
* <https://github.com/Impression2805/FMFP>
* <https://github.com/daintlab/confidence-aware-learning>

### Useful links
A list of papers that studies out-of-distribution (OOD) detection and misclassification detection (MisD)
* <https://github.com/Impression2805/Awesome-Failure-Detection>

### Contact for issues
Fei Zhu (zhufei2018@ia.ac.cn)
