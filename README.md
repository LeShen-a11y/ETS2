# ETSS
<div align="center">
<p align="center">
</p>

</div>

> **Event-Triggered Sparse Synchronization: An RGB-Efficient Hybrid Framework of Spiking Neural Network for Video Anomaly Detection**
> Le Shen*, Ling Luo*, Chong Wang, Wenxiu Huang, Shuhan Ye, Yuanbin Qian, Bolin Zhang,Jiangbo Qian

### :dart: Abstract
Event cameras have been introduced to video anomaly detection for their high dynamic range and temporal resolution, and combined with spiking neural networks (SNNs) they enable low-power, low-latency inference. However, SNN-based methods that rely only on sparse event streams often lack contextual semantics, limiting detection accuracy. Mean-while, most RGB–event fusion approaches introduce dense RGB computation at inference, leading to substantial ANN overhead and weakening the efficiency advantages of spike-driven SNN frameworks. To address this, we propose an efficient RGB–event hybrid SNN framework that leverages RGB and event streams
while using only a small set of key RGB clips to enhance semantics. Specifically, an Event-Triggered Sparse Sampling(ETSS) module selects dynamic RGB clips based on event saliency, and a Membrane Feature Injection (MFI) module injects RGB semantic features into event representations via the membrane mechanism for efficient fusion. Experiments on UCF-Crime-CEP demonstrate that the proposed framework approaches the performance of ANN-based counterparts while reducing inference energy consumption by 99.61%, demonstrating a favorable trade-off between detection performance and energy efficiency.
Index Terms—event camera, spiking neural network, video anomaly detection, RGB-event fusion
### :fire: What's New

### :gem: Framework
<img width="1849" height="883" alt="main_final" src="https://github.com/user-attachments/assets/b14a6ded-387a-4fff-b04f-d7a6438b1a36" />
Overview of the proposed RGB-sparse hybrid spiking framework for weakly supervised video anomaly detection. 

Given temporally aligned event frames and RGB clips, Event-Triggered Sampling (ETS) first computes event saliency scores via Event Saliency Calculation (ESC), and then applies
Gaussian-Mixture-Prior-based Local Region Partition (GMP) to divide the video into local motion-state regions. Based on these regions, region-aware sparse
sampling selects sparse yet representative RGB clips, which are encoded by I3D to extract semantic features. The feature filling module restores the sparse
RGB features into a continuous temporal representation for subsequent spiking processing and cross-modal interaction. In parallel, the event stream is
encoded by SpikingFormer to obtain event-driven temporal representations. Membrane Feature Injection (MFI) integrates the filled RGB semantics with event
representations at the membrane-feature level, and the fused temporal features are fed into a binary classifier to generate anomaly scores.

### 🔨: Training
Requirements: CUDA; numpy; tqdm; torchvision; timm==0.6.12; cupy==11.4.0; torch==1.12.1; spikingjelly==0.0.0.0.12;

The precomputed region labels and RGB selection flags for the **UCF-Crime dataset** under the best-performing RGB sparsity setting of **77.46%** are provided in: `ETSS/event_saliency/event_sparsity_77_46%`. In each file, `dynamic_label` denotes the motion-state region label, where `0`, `1`, and `2` correspond to strong-dynamic, weak-dynamic, and static regions, respectively. The `select_flag` column indicates whether the corresponding RGB clip is selected for RGB semantic extraction, where `1` denotes selected and other values denote not selected.

```
python main.py
```

### :memo: Supplementary Details

This section provides additional implementation details and experimental analyses.


#### Region-Aware RGB Budget Allocation

To achieve efficient RGB enhancement, ETS² performs region-aware RGB clip selection according to event saliency. 
The sampling budget is adaptively allocated among strong-dynamic, weak-dynamic, and static regions.

For example, in the UCF-Crime dataset, given a video with 171 candidate RGB clips and an RGB sparsity ratio of 80%, the retained RGB budget is 35 clips. 
Following the predefined allocation ratio of 2:2:1 (strong-dynamic : weak-dynamic : static), the budget is assigned as 14, 14, and 7 clips, respectively.

When the allocated budget exceeds the available candidates in a region, a capacity-aware adjustment is applied to redistribute the remaining clips.


#### Energy Evaluation

The proposed framework contains both ANN-based RGB processing and spike-driven SNN processing.
The inference energy is estimated by separately accounting for the two branches.

For ANN computation, energy consumption is estimated according to MAC operations:

\[
E_{ANN}=E_{MAC}\times \sum_l FLOP_l^{ANN}
\]

For the SNN branch, the energy is estimated based on synaptic operations (SOPs) considering spike rates and timesteps:

\[
SOP_l=R_l\times T\times FLOP_l
\]

The total inference energy is calculated as:

\[
E_{total}=E_{ANN}+E_{SNN}
\]

The energy evaluation follows the 45 nm CMOS hardware model, where MAC and AC operations consume 4.6 pJ and 0.9 pJ, respectively.


#### Event Partition and Timestep Analysis

We investigate the influence of event-frame granularity and spiking timesteps.

The best configuration is achieved with:

- temporal ratio: $\rho=4$
- spiking timestep: $T_{sf}=4$

which achieves an AUC of 82.86% on UCF-Crime-CEP.


#### Saliency Granularity Analysis

We compare two saliency estimation strategies:

- Event-frame-level estimation
- RGB-clip-level estimation

RGB-clip-level estimation achieves better performance because the saliency calculation scale is consistent with the RGB sampling unit.

The results are:

| Saliency Granularity | AUC (%) |
|---|---|
| Event-frame level | 82.53 |
| RGB-clip level | 82.86 |


#### Human Action Recognition Generalization

To evaluate the generalization capability of ETS² beyond anomaly detection, we further conduct experiments on HMDB51-CEP.

Following previous work, HMDB51 is paired with HMDB51-DVS to evaluate cross-modal event-RGB learning for fine-grained action recognition.

The event branch uses SpikingFormer-2-256 and the RGB branch uses ResNet-50.


### 💘 Acknowledgements
We thank the [SpikingJelly](https://github.com/fangwei123456/spikingjelly), [Spikingformer](https://github.com/zhouchenlin2096/Spikingformer) and [AR-Net](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020) for a quickly implement.
