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

## 📖 Supplementary Details

This section provides additional details about the proposed ETS² framework, including region-aware RGB budget allocation, energy evaluation, ablation studies, and generalization experiments.

---

### 1. Region-Aware RGB Budget Allocation

To illustrate the budget assignment strategy described in the main paper, we provide a detailed walkthrough using the **Abuse001** video from the UCF-Crime dataset.

This video contains \(N=171\) candidate RGB clips, with the following candidate counts:

| Region | Number of Candidate RGB Clips |
|---|---:|
| Strong-dynamic | 35 |
| Weak-dynamic | 33 |
| Static | 103 |
| **Total** | **171** |

The preset allocation ratio is **Strong-dynamic : Weak-dynamic : Static = 2 : 2 : 1**, reflecting the priority that dynamic regions are more likely to contain anomaly-related activities.

#### Example: RGB Sparsity Ratio of 80%

When the RGB sparsity ratio is

$$
r_{\mathrm{sp}}=0.8,
$$

the total sampling budget is

$$
n=171-\left\lfloor0.8\times171\right\rfloor=35.
$$

The preset ratio assigns 14, 14, and 7 clips to the strong-dynamic, weak-dynamic, and static regions, respectively.

| Region | Preset Allocation | Final Allocation |
|---|---:|---:|
| Strong-dynamic | 14 | 14 |
| Weak-dynamic | 14 | 14 |
| Static | 7 | 7 |

Since all allocated budgets are smaller than the corresponding region capacities, the final allocation remains unchanged.

#### Capacity-Aware Adjustment

When more RGB clips are retained, capacity adjustment may be required.

For example, retaining 85 clips gives an initial allocation of 34, 34, and 17 clips. However, the weak-dynamic region contains only 33 candidates. Its budget is therefore capped at 33, and the remaining clip is assigned to the static region, resulting in a final allocation of 34, 33, and 18.

A more capacity-limited case occurs when 100 clips are retained. The preset ratio gives 40, 40, and 20 clips, but the strong-dynamic and weak-dynamic regions contain only 35 and 33 candidates, respectively. After capping these two regions, the remaining budget is assigned to the static region, yielding a final allocation of 35, 33, and 32.

The preset ratio defines the preferred regional allocation, while the capacity-aware adjustment ensures that the final allocation satisfies the number of available candidates in each video.

#### Summary of Budget Allocation Scenarios

| Retained RGB Clips | Preset Strong | Preset Weak | Preset Static | Final Strong | Final Weak | Final Static |
|---:|---:|---:|---:|---:|---:|---:|
| 35 | 14 | 14 | 7 | 14 | 14 | 7 |
| 85 | 34 | 34 | 17 | 34 | 33 | 18 |
| 100 | 40 | 40 | 20 | 35 | 33 | 32 |

---

### 2. Energy Evaluation

Energy consumption is a crucial metric for evaluating neural architectures.

Since the proposed framework contains both dense ANN computation and spike-driven SNN computation, we estimate the theoretical inference energy by separately accounting for the ANN branch and the SNN branch.

Under a 45 nm CMOS hardware platform, the energy costs of multiply-accumulate and accumulate operations are:

| Operation | Energy Consumption |
|---|---:|
| MAC | 4.6 pJ |
| AC | 0.9 pJ |

#### ANN Branch

The computation of the ANN branch is dominated by multiply-accumulate (MAC) operations. Its energy consumption is estimated by accumulating the MAC-equivalent operation counts over all ANN layers:

<p align="center">
  <code>E<sub>ANN</sub> = E<sub>MAC</sub> × Σ<sub>k=1</sub><sup>L<sub>ANN</sub></sup> FLOP<sup>ANN</sup><sub>k</sub></code>
</p>

where:

- <i>L</i><sub>ANN</sub> denotes the number of layers in the ANN branch.
- FLOP<sup>ANN</sup><sub>k</sub> represents the number of floating-point operations in the <i>k</i>-th ANN layer.

For sparse inference, the layer-wise operation counts are calculated according to the actual number of selected RGB clips and the adopted crop setting.

---

#### SNN Branch

For the SNN branch, the number of synaptic operations in the <i>k</i>-th layer is estimated as:

<p align="center">
  <code>SOP<sub>k</sub> = R<sub>k</sub> × T × FLOP<sub>k</sub></code>
</p>

where:

- <i>R</i><sub>k</sub> ∈ [0, 1] is the average spike rate of layer <i>k</i>.
- <i>T</i> is the number of spiking timesteps.
- FLOP<sub>k</sub> is the number of floating-point operations in the corresponding non-spiking layer.

The energy consumption of the SNN branch is estimated as:

<p align="center">
  <code>E<sub>SNN</sub> = E<sub>MAC</sub> × FLOP<sub>1</sub> + E<sub>AC</sub> × Σ<sub>k=2</sub><sup>L<sub>SNN</sub></sup> SOP<sub>k</sub></code>
</p>

Here, FLOP<sub>1</sub> represents the number of floating-point operations in the first convolutional layer. For all subsequent layers with <i>k</i> ≥ 2, spike-driven binary activations are employed, and the computations are modeled as synaptic operations.

---

#### Total Inference Energy

The total inference energy of the proposed hybrid framework is:

<p align="center">
  <code>E<sub>total</sub> = E<sub>ANN</sub> + E<sub>SNN</sub></code>
</p>

---

### 3. Event Partition and Spiking Timestep Ablation

We evaluate the effects of event-frame granularity and spiking timesteps.

The temporal ratio factor ρ denotes the number of event frames corresponding to each RGB clip. The duration indicates the temporal aggregation window used to construct one event frame, and <i>T</i><sub>sf</sub> denotes the number of spiking timesteps in the SpikingFormer backbone of the event branch.

For ρ = 1 and <i>T</i><sub>sf</sub> = 1, events across a complete RGB clip, consisting of 16 RGB frames and approximately 533 ms at 30 FPS, are accumulated into a single event frame. This event frame is processed using one spiking timestep to produce the clip-level event representation.

For ρ = 2 and ρ = 4, the same 533 ms interval is divided into two and four shorter windows of approximately 266 ms and 133 ms, respectively. The resulting event frames provide finer temporal resolution and are processed over multiple spiking timesteps. Their outputs are averaged to obtain clip-level representations.

| ρ | Duration (ms) | T<sub>sf</sub> | AUC (%) |
|---:|---:|---:|---:|
| 1 | 533 | 1 | 81.81 |
| 2 | 266 | 2 | 78.91 |
| 4 | 133 | 4 | **82.86** |

Among all configurations, **ρ = 4 and T<sub>sf</sub> = 4** achieve the best AUC of **82.86%**.

The ρ = 1, T<sub>sf</sub> = 1 configuration accumulates events at the RGB-clip level, which may obscure intra-clip temporal variations. Moreover, single-timestep processing limits membrane-potential accumulation and constrains representation quality.

Although the ρ = 2, T<sub>sf</sub> = 2 configuration introduces finer event partitioning, two spiking timesteps provide insufficient temporal depth for effective feature integration, resulting in the lowest AUC of 78.91%.

In contrast, ρ = 4 and T<sub>sf</sub> = 4 provide an effective balance. The 133 ms windows preserve salient motion patterns, while four spiking timesteps enable sufficient membrane-potential accumulation for robust spatiotemporal modeling.

---

### 4. Saliency Granularity Ablation

We evaluate the effect of saliency-estimation granularity on the **UCF-Crime-CEP dataset** using 10-crop data augmentation and a fixed RGB sparsity ratio of **77.46%**.

With **ρ = 4**, each RGB clip is temporally aligned with four consecutive event frames. We compare two saliency-estimation strategies.

#### Event-Frame-Level Estimation

Saliency scores and region labels are computed independently for each of the four event frames.

The clip-level region label is then determined by majority voting over the four frame-level labels. Ties are resolved according to the following priority:

<p align="center">
  <strong>Strong-dynamic → Weak-dynamic → Static</strong>
</p>

This strategy captures fine-grained temporal variations but may introduce noise caused by transient frame-level fluctuations.

---

#### RGB-Clip-Level Estimation

The four event frames are first aggregated into a clip-level event representation through temporal accumulation.

The saliency score and region label are then calculated directly from the aggregated representation at the same temporal scale as RGB sampling.

This strategy provides more stable saliency estimation by smoothing intra-clip variations.

---

#### Results and Analysis

| Saliency Granularity | AUC (%) |
|---|---:|
| Event-frame level | 82.53 |
| RGB-clip level | **82.86** |

RGB-clip-level estimation achieves an AUC of **82.86%**, outperforming event-frame-level estimation by **0.33 percentage points**.

The improvement results from better alignment between the saliency-computation scale and the RGB sampling unit. Since RGB clips are selected at the clip level, estimating saliency from the aggregated clip-level event representation more accurately captures the overall event response relevant to the current sampling decision.

In contrast, event-frame-level estimation introduces a temporal-scale mismatch through majority voting, where transient frame-level variations may not reflect the actual saliency of the RGB clip as a whole.

These results indicate that RGB-clip-level saliency estimation provides more appropriate region labels for region-aware RGB selection.

---

### 5. Human Action Recognition Details

To evaluate the generalization capability of ETS² beyond binary video anomaly detection, we conduct experiments on **HMDB51-CEP**.

#### Dataset

Following previous work, HMDB51 is paired with HMDB51-DVS.

HMDB51 contains **6,766 RGB video clips from 51 action categories**. HMDB51-DVS is generated using the DAVIS240 event simulator with strict temporal synchronization.

This paired setting enables us to examine whether the proposed cross-modal design generalizes from binary anomaly detection to fine-grained multi-class action recognition.

---

#### Implementation

The prediction head and training objective are adapted to multi-class classification.

| Component | Backbone |
|---|---|
| Event branch | SpikingFormer-2-256 |
| RGB branch | ResNet-50 |

---

#### Evaluation Metric

Following existing action-recognition studies, top-1 accuracy on HMDB51-CEP is used as the primary metric for evaluating the discriminative capability of the proposed event-triggered sparse synchronization framework for action classification.


### 💘 Acknowledgements
We thank the [SpikingJelly](https://github.com/fangwei123456/spikingjelly), [Spikingformer](https://github.com/zhouchenlin2096/Spikingformer) and [AR-Net](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020) for a quickly implement.
