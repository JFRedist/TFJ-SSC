import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim

from sklearn.metrics import accuracy_score, f1_score

from model import Transformer
from early_stop_tool import EarlyStopping
from data_loader import load_dataset, get_folds, iterate_batches
from args import Config, Path


def set_random_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)        # CPU
    torch.cuda.manual_seed(seed)   # GPU


def autocast(config):
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=config.use_amp and config.device.type == 'cuda')


def test(model, dataset, labels, idx, config):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    pred = []
    label = []
    test_loss = 0

    with torch.no_grad(), autocast(config):
        for data, target in iterate_batches(dataset, labels, idx, config.batch_size):
            data = data.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)

            output = model(data)
            test_loss += criterion(output.float(), target).item()

            pred.append(torch.argmax(output, dim=1))
            label.append(target)

    # 最后一次性将结果转CPU计算
    pred = torch.cat(pred).cpu().numpy()
    label = torch.cat(label).cpu().numpy()
    accuracy = accuracy_score(label, pred, normalize=True, sample_weight=None)
    f1 = f1_score(label, pred, average='macro')

    return accuracy, test_loss, f1


def train(config, folds_to_run=None, save_all_checkpoint=False):
    path = Path()

    dataset, labels, subjects = load_dataset(path_labels=path.path_labels, path_dataset=path.path_TF)
    folds = get_folds(labels, subjects, config)
    print('cv_mode:', config.cv_mode, '| folds:', len(folds))

    for fold, split in enumerate(folds):
        if folds_to_run is not None and fold not in folds_to_run:
            continue
        print('\n', '-' * 15, '>', f'Fold {fold}', '<', '-' * 15)
        fold_dir = './Kfold_models/fold{}'.format(fold)
        os.makedirs(fold_dir, exist_ok=True)
        np.savez(os.path.join(fold_dir, 'split.npz'), **split)

        train_idx, val_idx, test_idx = split['train'], split['val'], split['test']
        print('train / val / test epochs: %d / %d / %d' % (len(train_idx), len(val_idx), len(test_idx)),
              '| subjects: %d / %d / %d' % tuple(len(np.unique(subjects[i])) for i in (train_idx, val_idx, test_idx)))

        set_random_seed(config.seed + fold)
        generator = torch.Generator().manual_seed(config.seed + fold)

        model = Transformer(config)
        model = model.to(config.device)

        criterion = nn.CrossEntropyLoss()

        # AdamW optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

        # 初始化当前学习率
        current_lr = config.learning_rate

        # 定义基于早停计数器的学习率调整函数
        def adjust_learning_rate(optimizer, early_stopping, epoch, initial_lr=config.learning_rate):
            """根据早停计数器状态调整学习率，模拟余弦退火效果"""
            # 获取早停计数器的值
            counter = early_stopping.counter
            patience = early_stopping.patience

            # 计算余弦衰减因子 (0到1之间)
            # 当counter增加时，衰减因子减小，学习率降低
            decay_factor = 0.5 * (1 + np.cos(np.pi * 2* min(counter, patience) / patience))

            # 计算新的学习率
            new_lr = config.lr_scheduler_eta_min + (initial_lr - config.lr_scheduler_eta_min) * decay_factor

            # 更新优化器中的学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            return new_lr

        # apply early_stop. If you want to view the full training process, set the save_all_checkpoint True
        early_stopping = EarlyStopping(patience=10, verbose=True, save_all_checkpoint=save_all_checkpoint)

        # evaluating indicator
        train_ACC = []
        train_LOSS = []
        test_ACC = []
        test_LOSS = []
        val_ACC = []
        val_LOSS = []

        n_batches = (len(train_idx) + config.batch_size - 1) // config.batch_size
        for epoch in range(config.num_epochs):
            t0 = time.time()
            running_loss = 0.0
            correct = 0

            model.train()

            # 每个epoch重新打乱训练集
            loop = tqdm(iterate_batches(dataset, labels, train_idx, config.batch_size, shuffle=True, generator=generator), total=n_batches)
            for data, target in loop:
                data = data.to(config.device, non_blocking=True)
                target = target.to(config.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast(config):
                    output = model(data)
                    loss = criterion(output.float(), target)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                batch_correct = torch.sum(torch.argmax(output, dim=1) == target).item()
                loop.set_postfix(train_acc=batch_correct / target.size(0), loss=loss.item())
                correct += batch_correct

            train_acc = correct / len(train_idx)
            test_acc, test_loss, test_f1 = test(model, dataset, labels, test_idx, config)
            val_acc, val_loss, val_f1 = test(model, dataset, labels, val_idx, config)

            # 检查早停计数器并更新学习率
            current_lr = adjust_learning_rate(optimizer, early_stopping, epoch)

            print('Epoch: ', epoch,
                  '| train loss: %.4f' % running_loss, '| train acc: %.4f' % train_acc,
                  '| val acc: %.4f' % val_acc, '| val loss: %.4f' % val_loss, '| val f1: %.4f' % val_f1,
                  '| test acc: %.4f' % test_acc, '| test loss: %.4f' % test_loss, '| test f1: %.4f' % test_f1,
                  '| lr: %.7f' % current_lr, '| %.0fs' % (time.time() - t0))

            train_ACC.append(train_acc)
            train_LOSS.append(running_loss)
            test_ACC.append(test_acc)
            test_LOSS.append(test_loss)
            val_ACC.append(val_acc)
            val_LOSS.append(val_loss)

            # Check whether to continue training. If save_all_checkpoint=False, the model name will be 'model.pkl'
            # 使用F1分数作为早停判断标准，而不是验证准确率
            early_stopping(val_f1, model, path=os.path.join(fold_dir, 'model_{}_epoch{}.pkl'.format(fold, epoch)))

            if early_stopping.early_stop:
                print("Early stopping at epoch ", epoch)
                break

        for name, values in [('train_LOSS', train_LOSS), ('train_ACC', train_ACC), ('test_LOSS', test_LOSS),
                             ('test_ACC', test_ACC), ('val_LOSS', val_LOSS), ('val_ACC', val_ACC)]:
            np.save(os.path.join(fold_dir, name + '.npy'), np.array(values))

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', help='only run these folds (default: all)')
    parser.add_argument('--epochs', type=int, help='override Config.num_epochs')
    parser.add_argument('--cv-mode', choices=['subject', 'epoch'], help='override Config.cv_mode')
    parser.add_argument('--batch-size', type=int, help='override Config.batch_size')
    args = parser.parse_args()

    config = Config()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.cv_mode is not None:
        config.cv_mode = args.cv_mode
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    set_random_seed(config.seed)
    train(config, folds_to_run=args.folds, save_all_checkpoint=False)
