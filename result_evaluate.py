import argparse
import numpy as np
from tqdm import tqdm

from sklearn.metrics import recall_score, accuracy_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix

import torch

from model import Transformer
from data_loader import load_dataset, get_folds, iterate_batches
from args import Config, Path


def specificity(y_true, y_pred, n=5):
    spec = []
    con_mat = confusion_matrix(y_true, y_pred)  # Each row is the ground truth, and each column is the precision
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spec1 = tn / (tn + fp)
        spec.append(spec1)
    average_specificity = np.mean(spec)
    return average_specificity


def class_wise_evaluate(con_mat):
    """
    Calculate the class_wise result through the confusion matrix
    Rows: Wake, N1, N2, N3
    Columns: precision, recall, F1_ score
    """
    class_wise_mat = np.empty((5, 3))
    for i in range(5):
        precision = con_mat[i, i] / np.sum(con_mat[:, i])
        recall = con_mat[i, i] / np.sum(con_mat[i, :])
        F1_score = (2 * precision * recall) / (precision + recall)
        class_wise_mat[i, 0] = precision
        class_wise_mat[i, 1] = recall
        class_wise_mat[i, 2] = F1_score

    return class_wise_mat


def test(model, dataset, labels, idx, config):
    model.eval()

    pred = []
    label = []

    amp = torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=config.use_amp and config.device.type == 'cuda')
    with torch.no_grad(), amp:
        n_batches = (len(idx) + config.batch_size - 1) // config.batch_size
        loop = tqdm(iterate_batches(dataset, labels, idx, config.batch_size), total=n_batches)
        for data, target in loop:
            data = data.to(config.device)
            target = target.to(config.device)

            output = model(data)

            # 使用PyTorch计算预测结果
            pred_batch = torch.argmax(output, dim=1)
            pred.extend(pred_batch.cpu().numpy())
            label.extend(target.cpu().numpy())

        accuracy = accuracy_score(label, pred, normalize=True, sample_weight=None)
        cohens_kappa = cohen_kappa_score(label, pred)
        macro_f1 = f1_score(label, pred, average='macro')
        average_sensitivity = recall_score(label, pred, average="macro")  # sensitivity and recall are the same concept
        average_specificity = specificity(label, pred, n=5)

        print('ACC: %.4f' % accuracy, 'k: %.4f' % cohens_kappa, 'MF1: %.4f' % macro_f1,
              'Sens: %.4f' % average_sensitivity, 'Spec: %.4f' % average_specificity)

        con_mat = confusion_matrix(label, pred, labels=list(range(5)))

    return accuracy, cohens_kappa, macro_f1, average_sensitivity, average_specificity, con_mat


def evaluate(config, path, folds_to_run=None):
    dataset, labels, subjects = load_dataset(path_labels=path.path_labels, path_dataset=path.path_TF)
    folds = get_folds(labels, subjects, config)
    print('cv_mode:', config.cv_mode)

    ACC = 0
    Kappa = 0
    MF1 = 0
    Sens = 0
    Spec = 0
    Confusion_mat = np.zeros([5, 5])
    n_eval = 0

    for fold, split in enumerate(folds):
        if folds_to_run is not None and fold not in folds_to_run:
            continue
        print('-' * 15, '>', f'Fold {fold}', '<', '-' * 15)

        path_model = './Kfold_models/fold{}/model.pkl'.format(fold)

        # the split saved by the trainer must match the one recomputed here
        saved = np.load('./Kfold_models/fold{}/split.npz'.format(fold))
        assert np.array_equal(saved['test'], split['test']), 'fold split differs from training (check cv_mode / num_fold / seed)'
        test_idx = split['test']

        print('train_set: ', len(split['train']))
        print('test_set: ', len(test_idx))

        model = Transformer(config)
        model = model.to(config.device)
        model.load_state_dict(torch.load(path_model, map_location=config.device), strict=True)

        accuracy, cohens_kappa, macro_f1, average_sensitivity, average_specificity, con_mat = test(model, dataset, labels, test_idx, config)
        n_eval += 1

        ACC += accuracy
        Kappa += cohens_kappa
        MF1 += macro_f1
        Sens += average_sensitivity
        Spec += average_specificity

        Confusion_mat += con_mat

        del model

    ACC /= n_eval
    Kappa /= n_eval
    MF1 /= n_eval
    Sens /= n_eval
    Spec /= n_eval

    class_wise_result = class_wise_evaluate(Confusion_mat)

    return ACC, Kappa, MF1, Sens, Spec, Confusion_mat, class_wise_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', help='only evaluate these folds (default: all)')
    parser.add_argument('--cv-mode', choices=['subject', 'epoch'], help='override Config.cv_mode')
    args = parser.parse_args()

    config = Config()
    if args.cv_mode is not None:
        config.cv_mode = args.cv_mode
    path = Path()

    ACC, Kappa, MF1, Sens, Spec, Confusion_mat, class_wise_result = evaluate(config=config, path=path, folds_to_run=args.folds)

    print('ACC: ', ACC)
    print('Cohen\'s Kappa: ', Kappa)
    print('MF1: ', MF1)
    print('Sens: ', Sens)
    print('Spec: ', Spec)
    print('confusion_mat:')
    print(Confusion_mat)
    print('class_wise_result: ')
    print(class_wise_result)
