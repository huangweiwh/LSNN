#!/usr/bin/env python

import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    AverageMeter,
    compute_accuracy,
    create_logger,
    get_rank,
)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    update_config(config)
    config.freeze()
    return config


def evaluate(config, model, test_loader, loss_func, logger):
    device = torch.device(config.device)

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    # hw
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    start = time.time()

    pred_raw_all = []
    pred_prob_all = []
    pred_label_all = []
    with torch.no_grad():
        for data, targets in tqdm.tqdm(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = loss_func(outputs, targets)
            # hw
            acc1, acc5 = compute_accuracy(config,
                                          outputs,
                                          targets,
                                          augmentation=False,
                                          topk=(1,5))

            pred_raw_all.append(outputs.cpu().numpy())
            pred_prob_all.append(F.softmax(outputs, dim=1).cpu().numpy())

            _, preds = torch.max(outputs, dim=1)
            pred_label_all.append(preds.cpu().numpy())

            loss_ = loss.item()
            # hw
            acc1_ = acc1.item()
            acc5_ = acc5.item()

            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            # hw
            acc1_meter.update(acc1_, num)
            acc5_meter.update(acc5_, num)

            correct_meter.update(correct_, 1)

        accuracy = correct_meter.sum / len(test_loader.dataset)

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')
        logger.info(f'Loss {loss_meter.avg:.4f} Accuracy {accuracy:.4f}')
        logger.info(f'Acc@1 {acc1_meter.avg:.4f} Acc@5 {acc5_meter.avg:.4f}')

    preds = np.concatenate(pred_raw_all)
    probs = np.concatenate(pred_prob_all)
    labels = np.concatenate(pred_label_all)
    return preds, probs, labels, loss_meter.avg, accuracy, acc1_meter.avg, acc5_meter.avg


def main():
    config = load_config()

    if config.test.output_dir is None:
        output_dir = pathlib.Path(config.test.checkpoint).parent
    else:
        output_dir = pathlib.Path(config.test.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    logger = create_logger(name=__name__, distributed_rank=get_rank(), output_dir=output_dir,
                           filename='test.txt')

    model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)
    ## if Checkpointer has anything question, modify "checkpoint.py", comment the raise error, turn into "pass". It may be helpful.
    checkpointer = Checkpointer(model,
                                checkpoint_dir=output_dir,
                                logger=logger,
                                distributed_rank=get_rank())
    checkpointer.load(config.test.checkpoint)
    test_loader = create_dataloader(config, is_train=False)
    _, test_loss = create_loss(config)

    preds, probs, labels, loss, acc, acc_1, acc_5 = evaluate(config, model, test_loader,
                                                            test_loss, logger)

    output_path = output_dir / f'predictions.npz'
    np.savez(output_path,
             preds=preds,
             probs=probs,
             labels=labels,
             loss=loss,
             acc=acc,
             acc_1=acc_1,
             acc_5=acc_5)


if __name__ == '__main__':
    main()
