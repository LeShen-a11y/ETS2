from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import os

from dataset import Dataset_combine
from test import test
from train import  train

from utils import save_best_record

from model import Model


import option
from tqdm import tqdm
from config import Config


if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    # ---------- Data ----------
    train_nloader_com = DataLoader(Dataset_combine(args, test_mode=False, is_normal=True),
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=0, pin_memory=False, drop_last=True)

    train_aloader_com = DataLoader(Dataset_combine(args, test_mode=False, is_normal=False),
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset_combine(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)


    # model = Model(args.feature_size, args.batch_size)
    model = Model(
        F_dvs=256,
        F_rgb=2048,
        d_model=256,
        out_ch=1,
    )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    # print(f'Trainable : {trainable:,}')
    print(f'Total     : {total:,}')


    print('device:', device)

    # os.makedirs('../ckpt', exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)



    # ---------- Early stopping ----------
    best_AUC = -1
    patience_counter = 0
    test_info = {"epoch": [], "test_AUC": []}
    output_path = 'output'  # 自行修改

    loadern_iter = iter(train_nloader_com)
    loadera_iter = iter(train_aloader_com)


     # 加载权重
    # checkpoint = torch.load("ckpt/snn-mix_81_81.pkl",device)
    checkpoint = torch.load("/home/cy518/sl/code/ETSF/Select_UCF_Crime_CEP/ckpt/rgb_22_54/snn-mix_82_48.pkl",device)
    # 若 checkpoint 含 optimizer 等信息，取模型字典
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # # ---------- Training ----------
    # for step in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True):
    #
    #     # LR schedule
    #     if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
    #         for g in optimizer.param_groups:
    #             g['lr'] = config.lr[step - 1]
    #     # 重载迭代器
    #     if (step - 1) % len(train_nloader_com) == 0:
    #         loadern_iter = iter(train_nloader_com)
    #     if (step - 1) % len(train_aloader_com) == 0:
    #         loadera_iter = iter(train_aloader_com)
    #
    #
    #     train(loadern_iter, loadera_iter, model, args.batch_size,optimizer,  device)
    #
    #
    #     if step % 5 == 0 and step > 4:
    #         auc = test(test_loader, model, args, device,
    #                    gt_path='list/gt-ucf.npy',
    #                    save_output=False)
    #
    #         test_info["epoch"].append(step)
    #         test_info["test_AUC"].append(auc)
    #
    #
    #
    #         # 是否有提升
    #         if auc > best_AUC + args.early_stop_delta:
    #             best_AUC = auc
    #             patience_counter = 0
    #
    #             best_path = './ckpt/' + args.model_name + '_best.pkl'
    #             torch.save(model.state_dict(), best_path)
    #
    #             save_best_record(test_info,
    #                              os.path.join(output_path, f'{step}-iter-AUC.txt'))
    #
    #
    #             if not args.save_best_only:
    #                 torch.save(model.state_dict(),
    #                            f'./ckpt/{args.model_name}{step}{best_AUC}.pkl')
    #
    #         else:
    #             patience_counter += 1
    #
    #         # Early stop
    #         if patience_counter >= args.early_stop_patience:
    #             print(f"[Early-Stop] No improvement for {patience_counter} validations. "
    #                   f"Best AUC={best_AUC:.4f} at epoch {step}.")
    #             break

    auc = test(test_loader, model, args, device,
               gt_path='list/gt-ucf.npy',
               save_output=False)
    print("AUC:",auc)

    # ---------- 结束 ----------
    # print("best AUC:", best_AUC)



