"""Rank seq_trainer.py runs by validation macro-F1 only (test metrics are deliberately not shown).

For every run and fold, the score is the best validation macro-F1, i.e. the checkpoint early stopping kept.
Runs are compared on the folds they all share.

    python tune_report.py                      # all runs under runs/
    python tune_report.py --folds 0 1 2        # restrict to these folds
"""
import os
import json
import argparse
import numpy as np


def load_run(run_dir):
    scores = {}
    for name in sorted(os.listdir(run_dir)):
        h = os.path.join(run_dir, name, 'history.json')
        if name.startswith('fold') and os.path.exists(h):
            hist = json.load(open(h))
            best = int(np.argmax(hist['val_f1']))
            scores[int(name[4:])] = {'val_f1': hist['val_f1'][best], 'val_acc': hist['val_acc'][best],
                                     'best_epoch': best, 'epochs': len(hist['val_f1'])}
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--folds', type=int, nargs='+', help='only compare these folds')
    parser.add_argument('--include', nargs='+', help='only these run names')
    args = parser.parse_args()

    runs = {}
    for name in sorted(os.listdir(args.runs_dir)):
        if args.include and name not in args.include:
            continue
        cfg = os.path.join(args.runs_dir, name, 'config.json')
        if os.path.exists(cfg) and json.load(open(cfg)).get('num_fold') != '5':
            continue                                  # different fold definitions are not comparable
        scores = load_run(os.path.join(args.runs_dir, name))
        if scores:
            runs[name] = scores

    folds = set(args.folds) if args.folds else set.intersection(*(set(s) for s in runs.values()))
    rows = []
    for name, scores in runs.items():
        if not folds <= set(scores):
            continue
        f1 = [scores[k]['val_f1'] for k in sorted(folds)]
        acc = [scores[k]['val_acc'] for k in sorted(folds)]
        rows.append((np.mean(f1), name, f1, np.mean(acc), [scores[k]['best_epoch'] for k in sorted(folds)]))

    print('validation macro-F1 on folds', sorted(folds), '(test metrics not shown)\n')
    print('%-24s %8s %8s   %-28s %s' % ('run', 'val MF1', 'val Acc', 'per-fold val MF1', 'best epoch'))
    for mean_f1, name, f1, mean_acc, ep in sorted(rows, reverse=True):
        print('%-24s %8.2f %8.2f   %-28s %s' % (name, mean_f1 * 100, mean_acc * 100, ' '.join('%.2f' % (v * 100) for v in f1), ep))
