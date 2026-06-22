# ETSF
<div align="center">
<p align="center">
</p>

</div>

> **Event-Triggered Sparse Fusion: An RGB-Sparse Hybrid Framework of Spiking Neural Network for Video Anomaly Detection**
> Le Shen*, Ling Luo*, Chong Wang, Wenxiu Huang, Shuhan Ye, Yuanbin Qian, Bolin Zhang,Jiangbo Qian

### :dart: Abstract
Event cameras have been introduced to video anomaly detection for their high dynamic range and temporal resolution, and combined with spiking neural networks (SNNs) they enable low-power, low-latency inference. However, SNN-based methods that rely only on sparse event streams often lack contextual semantics, limiting detection accuracy. Mean-while, most RGB–event fusion approaches introduce dense RGB computation at inference, leading to substantial ANN overhead and weakening the efficiency advantages of spike-driven SNN frameworks. To address this, we propose an efficient RGB–event hybrid SNN framework that leverages RGB and DVS signals
while using only a small set of key RGB clips to enhance semantics. Specifically, an Event-Triggered Sparse Sampling(ETSS) module selects dynamic RGB clips based on event saliency, and a Membrane Feature Injection (MFI) module injects RGB semantic features into event representations via the membrane mechanism for efficient fusion. Experiments on UCF-Crime-CEP demonstrate that the proposed framework approaches the performance of ANN-based counterparts while reducing inference energy consumption by 99.51%, demonstrating a favorable trade-off between detection performance and energy efficiency.
Index Terms—event camera, spiking neural network, video anomaly detection, RGB-event fusion
### :fire: What's New

### :gem: Framework

<img width="1855" height="893" alt="main_final" src="https://github.com/user-attachments/assets/766a81ef-b285-482c-9039-d07451407e14" />
Overview of the proposed RGB-sparse hybrid spiking framework for weakly supervised video anomaly detection. Given temporally aligned event
frames and RGB clips, Event-Triggered Sparse Sampling (ETSS) first computes event saliency scores via Event Saliency Calculation (ESC), and then applies
Gaussian-Mixture-Prior-based Local Region Partition (GMP) to divide the video into local motion-state regions. Based on these regions, region-aware sparse
sampling selects sparse yet representative RGB clips, which are encoded by I3D to extract semantic features. The feature filling module restores the sparse
RGB features into a continuous temporal representation for subsequent spiking processing and cross-modal interaction. In parallel, the event stream is
encoded by SpikingFormer to obtain event-driven temporal representations. Membrane Feature Injection (MFI) integrates the filled RGB semantics with event
representations at the membrane-feature level, and the fused temporal features are fed into a binary classifier to generate anomaly scores.

### 🔨: Training
Requirements: CUDA; numpy; tqdm; torchvision; timm==0.6.12; cupy==11.4.0; torch==1.12.1; spikingjelly==0.0.0.0.12;
```
python main.py
```

### 💘 Acknowledgements
We thank the [SpikingJelly](https://github.com/fangwei123456/spikingjelly), [Spikingformer](https://github.com/zhouchenlin2096/Spikingformer) and [AR-Net](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020) for a quickly implement.
