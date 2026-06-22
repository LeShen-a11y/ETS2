import numpy as np
import torch
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from spikingjelly.clock_driven import functional


def find_best_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    f1_scores = np.nan_to_num(f1_scores)

    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx]


def calculate_far(y_true, y_score, threshold):
    y_pred = (y_score > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn) if fp + tn > 0 else 0.0
    return far


def process_logits(logits):
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = torch.sigmoid(logits)

    if logits.dim() == 4:
        if logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        logits = logits.mean(dim=1)

    elif logits.dim() == 3:
        if logits.size(-1) == 1:
            logits = logits.squeeze(-1)

    if logits.dim() == 3 and logits.size(1) > 1:
        logits = logits.mean(dim=1)

    return logits


def test(
    dataloader,
    model,
    args,
    device,
    gt_path="list/gt-ucf.npy",
    save_output=False,
    output_dir="output",
):
    model.eval()
    all_preds = []

    with torch.no_grad():
        try:
            gt = np.load(gt_path)
        except Exception as e:
            print(f"加载真实标签失败: {e}")
            return 0.0

        for inputs in dataloader:
            inputs = inputs.to(device)

            if inputs.shape[-1] > 2048:
                input_rgb = inputs[:, :, :, :2048]
                input_dvs = inputs[:, :, :, 2048:]
            else:
                input_rgb = None
                input_dvs = inputs

            functional.reset_net(model)

            output = model(inputs_dvs=input_dvs, inputs_rgb=input_rgb)
            logits = process_logits(output)

            batch_preds = [
                logits[b].detach().cpu().numpy()
                for b in range(logits.size(0))
            ]

            all_preds.extend(batch_preds)

        if not all_preds:
            print("没有预测结果！")
            return 0.0

        pred_segments = np.concatenate(all_preds, axis=0)
        pred_frames = np.repeat(pred_segments, 16)

        if len(pred_frames) != len(gt):
            min_len = min(len(pred_frames), len(gt))
            pred_frames = pred_frames[:min_len]
            gt = gt[:min_len]

        y_true = gt.astype(np.int32)
        y_score = pred_frames.astype(np.float32)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        print(f"AUC: {roc_auc:.4f}")

        return roc_auc