import torch
from spikingjelly.clock_driven import functional


def sparsity_loss(scores, lamda2):
    return lamda2 * torch.mean(scores)


def smooth_loss(scores, lamda1):
    diff = scores[:, 1:] - scores[:, :-1]
    return lamda1 * torch.mean(diff ** 2)


def to_2B_T_logits(scores):
    x = scores

    if x.dim() == 4:
        if x.size(-1) == 1:
            x = x.squeeze(-1)
        x = x.mean(dim=1)

    elif x.dim() == 3:
        if x.size(-1) == 1:
            x = x.squeeze(-1)
        else:
            x = x.mean(dim=1)

    elif x.dim() == 2:
        pass

    else:
        raise ValueError(f"Unsupported scores shape: {scores.shape}")

    return x


def BCE_loss_binary_mil(outputs, labels, bce=None, k_ratio=0.09375):
    logits = to_2B_T_logits(outputs)
    labels = labels.float().view(-1)

    if bce is None:
        bce = torch.nn.BCEWithLogitsLoss()

    _, T = logits.shape
    k = max(1, int(T * k_ratio))

    topk_logits, _ = torch.topk(logits, k=k, dim=1)
    video_logits = topk_logits.mean(dim=1)

    loss_bce = bce(video_logits, labels)
    frame_scores = torch.sigmoid(logits)

    return loss_bce, video_logits, frame_scores


def train(
    nloader,
    aloader,
    model,
    batch_size,
    optimizer,
    device,
    lam_bce=1,
    lam_sparse=8e-3,
    lam_smooth=8e-4,
    k_ratio=0.09375,
):
    model.train()

    try:
        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)
    except StopIteration:
        return None

    ninput = ninput.to(device)
    nlabel = nlabel.to(device)
    ainput = ainput.to(device)
    alabel = alabel.to(device)

    x = torch.cat((ninput, ainput), dim=0)
    labels_2B = torch.cat((nlabel, alabel), dim=0).float().view(-1)

    if x.shape[-1] > 2048:
        x_rgb = x[:, :, :, :2048]
        x_dvs = x[:, :, :, 2048:]
    else:
        x_rgb = x
        x_dvs = x

    output = model(inputs_dvs=x_dvs, inputs_rgb=x_rgb)
    logits = output[0] if isinstance(output, tuple) else output
    logits_2B_T = to_2B_T_logits(logits)

    loss_bce, video_logits, frame_scores = BCE_loss_binary_mil(
        outputs=logits_2B_T,
        labels=labels_2B,
        bce=torch.nn.BCEWithLogitsLoss(),
        k_ratio=k_ratio,
    )

    abn_scores = frame_scores[batch_size:, :]

    loss_sparse = sparsity_loss(abn_scores, lam_sparse)
    loss_smooth = smooth_loss(abn_scores, lam_smooth)

    cost = lam_bce * loss_bce + loss_sparse + loss_smooth

    optimizer.zero_grad()
    cost.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    functional.reset_net(model)

    print(f"  Total Loss: {cost.item():.6f}")
    print(f"  BCE Loss: {loss_bce.item():.6f} (λ={lam_bce})")
    print(f"  Sparse Loss: {loss_sparse.item():.6f} (λ={lam_sparse})")
    print(f"  Smooth Loss: {loss_smooth.item():.6f} (λ={lam_smooth})")

    return None