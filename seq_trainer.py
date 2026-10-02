"""Train and evaluate SeqTFJSSC (slim epoch encoder + inter-epoch Transformer) with subject-wise k-fold CV.

Uses the same subject-wise folds, early stopping on validation macro-F1 and counter-based learning-rate
schedule as Kfold_trainer.py, so results are directly comparable with the baseline.

Examples:
    python seq_trainer.py --run-name A_slim_L1 --seq-len 1
    python seq_trainer.py --run-name B_slim_L15 --seq-len 15
    python seq_trainer.py --run-name C_slim_L15_w --seq-len 15 --class-weighting inv_sqrt
    python seq_trainer.py --run-name C_slim_L15_w --summarize      # recompute metrics from saved predictions
"""
import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, recall_score, confusion_matrix

from model_seq import SeqTFJSSC
from early_stop_tool import EarlyStopping
from data_loader import load_dataset, get_folds, record_ranges, sequence_starts, iterate_sequences
from result_evaluate import specificity, class_wise_evaluate
from args import SeqConfig, Path

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def set_random_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def autocast(config):
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=config.use_amp and config.device.type == 'cuda')


def class_weights(y, mode, num_classes):
    if mode == 'none':
        return None
    freq = np.bincount(y, minlength=num_classes) / len(y)
    if mode == 'inv_sqrt':
        w = 1 / np.sqrt(freq)
    else:
        raise ValueError(f'unknown class_weighting: {mode}')
    w = w / np.sum(w * freq)                 # expected weight per sample = 1
    return torch.tensor(w, dtype=torch.float32)


def predict(model, dataset, labels, ranges, config):
    """Sliding-window prediction; overlapping softmax outputs of each epoch are averaged."""
    model.eval()
    starts = sequence_starts(ranges, config.seq_len, stride=min(config.seq_eval_stride, config.seq_len))
    prob_sum = torch.zeros(len(labels), config.num_classes)
    count = torch.zeros(len(labels))
    with torch.no_grad(), autocast(config):
        for data, _, index in iterate_sequences(dataset, labels, starts, config.seq_len, config.seq_batch_size * 2):
            probs = torch.softmax(model(data.to(config.device, non_blocking=True)).float(), dim=-1).cpu()
            prob_sum.index_add_(0, index.flatten(), probs.view(-1, config.num_classes))
            count.index_add_(0, index.flatten(), torch.ones(index.numel()))
    idx = np.concatenate([np.arange(s, e) for s, e in ranges])
    assert (count[idx] > 0).all(), 'some epochs were not covered by any window'
    probs = (prob_sum[idx] / count[idx, None]).numpy()
    y = labels[idx].numpy()
    loss = float(-np.log(probs[np.arange(len(y)), y] + 1e-8).mean())
    return y, probs.argmax(axis=1), probs, idx, loss


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, early_stopping, config):
    """根据早停计数器状态调整学习率，模拟余弦退火效果 (same schedule as Kfold_trainer.py)"""
    counter, patience = early_stopping.counter, early_stopping.patience
    decay_factor = 0.5 * (1 + np.cos(np.pi * 2 * min(counter, patience) / patience))
    new_lr = config.lr_scheduler_eta_min + (config.learning_rate - config.lr_scheduler_eta_min) * decay_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def train_fold(fold, split, dataset, labels, records, subjects, config, run_dir):
    fold_dir = os.path.join(run_dir, f'fold{fold}')
    os.makedirs(fold_dir, exist_ok=True)
    np.savez(os.path.join(fold_dir, 'split.npz'), **split)
    ranges = {k: record_ranges(split[k], records) for k in ('train', 'val', 'test')}
    print('train / val / test epochs: %d / %d / %d' % tuple(len(split[k]) for k in ('train', 'val', 'test')),
          '| subjects: %d / %d / %d' % tuple(len(np.unique(subjects[split[k]])) for k in ('train', 'val', 'test')),
          '| recordings: %d / %d / %d' % tuple(len(ranges[k]) for k in ('train', 'val', 'test')))

    set_random_seed(config.seed + fold)
    generator = torch.Generator().manual_seed(config.seed + fold)

    model = SeqTFJSSC(config).to(config.device)
    weight = class_weights(labels[split['train']].numpy(), config.class_weighting, config.num_classes)
    if weight is not None:
        print('class weights:', dict(zip(STAGES, np.round(weight.numpy(), 3))))
        weight = weight.to(config.device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    # cosine runs its full length; early stopping then only selects the best checkpoint
    cosine = config.lr_schedule == 'cosine'
    early_stopping = EarlyStopping(patience=config.cosine_epochs if cosine else 10, verbose=True)
    num_epochs = min(config.num_epochs, config.cosine_epochs) if cosine else config.num_epochs
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'test_acc', 'test_f1', 'lr')}
    global_step = 0
    current_lr = config.learning_rate

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        starts = sequence_starts(ranges['train'], config.seq_len, generator=generator)
        n_batches = (len(starts) + config.seq_batch_size - 1) // config.seq_batch_size
        warmup_steps = config.warmup_epochs * n_batches
        running_loss, correct, seen = 0.0, 0, 0
        loop = tqdm(iterate_sequences(dataset, labels, starts, config.seq_len, config.seq_batch_size, shuffle=True,
                                      generator=generator), total=n_batches)
        for data, target, _ in loop:
            data = data.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(config):
                output = model(data)
                loss = criterion(output.float().reshape(-1, config.num_classes), target.reshape(-1))
            loss.backward()
            if global_step < warmup_steps:
                current_lr = config.learning_rate * (global_step + 1) / warmup_steps
                set_lr(optimizer, current_lr)
            optimizer.step()
            global_step += 1
            running_loss += loss.item()
            batch_correct = (output.argmax(dim=-1) == target).sum().item()
            correct += batch_correct
            seen += target.numel()
            loop.set_postfix(train_acc=batch_correct / target.numel(), loss=loss.item())

        y_val, p_val, _, _, val_loss = predict(model, dataset, labels, ranges['val'], config)
        y_test, p_test, _, _, _ = predict(model, dataset, labels, ranges['test'], config)
        val_acc, val_f1 = accuracy_score(y_val, p_val), f1_score(y_val, p_val, average='macro')
        test_acc, test_f1 = accuracy_score(y_test, p_test), f1_score(y_test, p_test, average='macro')
        in_warmup = epoch + 1 < config.warmup_epochs
        if config.lr_schedule == 'restart' and not in_warmup:
            current_lr = adjust_learning_rate(optimizer, early_stopping, config)   # uses the counter before this epoch, as in the thesis

        early_stopping(val_f1, model, path=os.path.join(fold_dir, 'model.pkl'))

        if not in_warmup:
            if config.lr_schedule == 'plateau' and early_stopping.counter > 0 and early_stopping.counter % config.plateau_patience == 0:
                current_lr = max(current_lr * config.plateau_factor, config.lr_scheduler_eta_min)
                set_lr(optimizer, current_lr)
            elif cosine:
                t = min(1.0, (epoch + 1 - config.warmup_epochs) / max(1, config.cosine_epochs - config.warmup_epochs))
                current_lr = config.lr_scheduler_eta_min + (config.learning_rate - config.lr_scheduler_eta_min) * 0.5 * (1 + np.cos(np.pi * t))
                set_lr(optimizer, current_lr)

        print('Epoch: ', epoch, '| train loss: %.4f' % (running_loss / n_batches), '| train acc: %.4f' % (correct / seen),
              '| val acc: %.4f' % val_acc, '| val loss: %.4f' % val_loss, '| val f1: %.4f' % val_f1,
              '| test acc: %.4f' % test_acc, '| test f1: %.4f' % test_f1, '| lr: %.7f' % current_lr,
              '| %.0fs' % (time.time() - t0))
        for k, v in zip(history, (running_loss / n_batches, correct / seen, val_loss, val_acc, val_f1, test_acc, test_f1, current_lr)):
            history[k].append(v)

        if early_stopping.early_stop:
            print('Early stopping at epoch ', epoch)
            break

    json.dump(history, open(os.path.join(fold_dir, 'history.json'), 'w'))

    # test predictions of the best (validation macro-F1) checkpoint
    model.load_state_dict(torch.load(os.path.join(fold_dir, 'model.pkl'), map_location=config.device))
    y, pred, probs, idx, _ = predict(model, dataset, labels, ranges['test'], config)
    np.savez(os.path.join(fold_dir, 'test_predictions.npz'), y_true=y, y_pred=pred, probs=probs, index=idx)
    print('fold %d best checkpoint: test acc %.4f | test f1 %.4f' % (fold, accuracy_score(y, pred), f1_score(y, pred, average='macro')))
    del model
    torch.cuda.empty_cache()


def summarize(run_dir, num_fold):
    per_fold, pooled = [], np.zeros((5, 5))
    for fold in range(num_fold):
        f = os.path.join(run_dir, f'fold{fold}', 'test_predictions.npz')
        if not os.path.exists(f):
            continue
        d = np.load(f)
        y, p = d['y_true'], d['y_pred']
        per_fold.append({'fold': fold, 'acc': accuracy_score(y, p), 'kappa': cohen_kappa_score(y, p),
                         'mf1': f1_score(y, p, average='macro'), 'sens': recall_score(y, p, average='macro'),
                         'spec': specificity(y, p, n=5)})
        pooled += confusion_matrix(y, p, labels=list(range(5)))
    if not per_fold:
        print('no predictions found in', run_dir)
        return None
    metrics = {k: (float(np.mean([r[k] for r in per_fold])), float(np.std([r[k] for r in per_fold], ddof=1)) if len(per_fold) > 1 else 0.0)
               for k in ('acc', 'kappa', 'mf1', 'sens', 'spec')}
    class_wise = class_wise_evaluate(pooled)
    summary = {'folds': per_fold, 'mean_std': metrics, 'per_class_f1': dict(zip(STAGES, class_wise[:, 2].tolist())),
               'confusion_matrix': pooled.astype(int).tolist()}
    json.dump(summary, open(os.path.join(run_dir, 'summary.json'), 'w'), indent=2)

    print('\n==== %s (%d folds) ====' % (os.path.basename(run_dir), len(per_fold)))
    for r in per_fold:
        print('fold %d: acc %.4f | kappa %.4f | mf1 %.4f' % (r['fold'], r['acc'], r['kappa'], r['mf1']))
    print('ACC %.2f ± %.2f | kappa %.3f ± %.3f | MF1 %.2f ± %.2f' % (
        metrics['acc'][0] * 100, metrics['acc'][1] * 100, metrics['kappa'][0], metrics['kappa'][1],
        metrics['mf1'][0] * 100, metrics['mf1'][1] * 100))
    print('per-class F1:', ' | '.join('%s %.2f' % (s, v * 100) for s, v in summary['per_class_f1'].items()))
    print('confusion matrix (rows = truth):\n', pooled.astype(int))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True, help='results go to runs/<run-name>/')
    parser.add_argument('--seq-len', type=int, help='override SeqConfig.seq_len')
    parser.add_argument('--class-weighting', choices=['none', 'inv_sqrt'], help='override SeqConfig.class_weighting')
    parser.add_argument('--lr', type=float, help='override SeqConfig.learning_rate')
    parser.add_argument('--num-fold', type=int, help='override Config.num_fold (number of CV folds)')
    parser.add_argument('--folds', type=int, nargs='+', help='only run these folds (default: all)')
    parser.add_argument('--epochs', type=int, help='override SeqConfig.num_epochs')
    parser.add_argument('--set', nargs='+', action='extend', default=[], metavar='KEY=VALUE',
                        help='override any SeqConfig attribute, e.g. --set dropout=0.2 seq_num_encoder=4')
    parser.add_argument('--summarize', action='store_true', help='only recompute the summary from saved predictions')
    args = parser.parse_args()

    config = SeqConfig()
    for name, attr in (('num_fold', 'num_fold'), ('seq_len', 'seq_len'), ('class_weighting', 'class_weighting'), ('lr', 'learning_rate'), ('epochs', 'num_epochs')):
        if getattr(args, name) is not None:
            setattr(config, attr, getattr(args, name))
    for item in args.set:
        key, value = item.split('=', 1)
        assert hasattr(config, key), f'unknown config attribute: {key}'
        setattr(config, key, type(getattr(config, key))(value))
    config.seq_batch_size = max(1, config.epochs_per_batch // config.seq_len)
    assert config.cv_mode == 'subject', 'sequence windows need whole recordings per split (cv_mode subject)'
    run_dir = os.path.join('runs', args.run_name)

    if not args.summarize:
        os.makedirs(run_dir, exist_ok=True)
        json.dump({k: str(v) for k, v in vars(config).items()}, open(os.path.join(run_dir, 'config.json'), 'w'), indent=2)
        path = Path()
        dataset, labels, subjects, records = load_dataset(path.path_labels, path.path_TF, return_records=True)
        folds = get_folds(labels, subjects, config)
        n_params = sum(p.numel() for p in SeqTFJSSC(config).parameters())
        print('run:', args.run_name, '| seq_len:', config.seq_len, '| class_weighting:', config.class_weighting,
              '| lr:', config.learning_rate, '| params: %.2fM' % (n_params / 1e6))
        for fold, split in enumerate(folds):
            if args.folds is not None and fold not in args.folds:
                continue
            print('\n', '-' * 15, '>', f'Fold {fold}', '<', '-' * 15)
            train_fold(fold, split, dataset, labels, records, subjects, config, run_dir)

    summarize(run_dir, config.num_fold)
