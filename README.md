This repository contains code and baseline implementations related to our study **"High-Performance Cross-Subject Decoding of Multiclass Rhythmic Motor Imagery Using EEG Data from 100 Subjects"** published on IEEE TNSRE 2026.

## Overview

In this work, we mainly investigate **how data influences cross-subject decoding performance** in EEG-based BCI, rather than focusing only on increasingly complex model architectures.

Our findings suggest that very simple MLP-based networks can achieve results comparable to, or even better than, more sophisticated models on our proposed rhythmic MI paradigm. These results highlight the joint importance of **model design** and **data characteristics** in achieving robust cross-subject generalization.

More broadly, this work suggests that future progress in EEG-based BCI may benefit from shifting attention away from purely architectural innovation and toward the **coordinated optimization of model simplicity, dataset scale, and data quality**.

## Current Status of the Dataset

The self-collected rhythmic MI dataset used in this work is **still under construction and continuous refinement**, and it will be released in the future.

Nevertheless, we hope that the ideas, experimental framework, and baseline implementations provided here can still offer useful insights to the field and support future related research.

Also, if you are interested in conducting Rhythmic MI experiment your self (offline / online), there are more details in the following articles:

[1] Wei, Y., Meng, J., Luo, R., Mai, X., Li, S., Xia, Y., & Zhu, X. (2024). Action observation with rhythm imagery (AORI): a novel paradigm to activate motor-related pattern for high-performance motor decoding. IEEE Transactions on Biomedical Engineering, 72(3), 1085-1096.

[2] Wei, Y., Luo, R., Xia, Y., Mai, X., Zhu, X., & Meng, J. (2026). Rhythmic Motor Imagery Boosts Accuracy and Efficiency in Noninvasive Brain-Computer Interfaces. IEEE Journal of Biomedical and Health Informatics.

Please feel free to contact me if you have any questions about the experiment.

## What is Included in This Repository

This repository currently provides:

- a streamlined MLP-based framework for cross-subject decoding,
- reference implementations of representative baseline methods such as:
  - **EEGNet**
  - **LMDANet**
  - **FACTNet**
  - **SSVEPFormer**
  - and other related models

## Motivation

While many previous studies have focused primarily on designing more advanced neural architectures, our results suggest that:

- **simple models can be highly competitive** under suitable data conditions,
- **dataset scale and data characteristics** may play a central role in cross-subject generalization,
- and future improvements may come not only from better models, but also from better **data construction, curation, and quality control**.

## Contact

Please feel free to contact me at **stepfurther678@sjtu.edu.cn** for any questions or discussions.
