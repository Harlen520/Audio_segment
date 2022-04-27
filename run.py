import argparse
import os
import random
import time
import logging

import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.transformers import LinearDecayWithWarmup
from sklearn.metrics import precision_recall_fscore_support
from paddle.metric import Accuracy

from data import create_dataloader
from paddlenlp.datasets import load_dataset
from data_process.features_extract import read_data
from model import AudioSegmentation

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

id2label = {
    0: "no-music",
    1: "music   "
}

def getArgs():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str, default='/home/th/paddle/audio_segment/data/dataset/train', required=False, help="The full path of train_set_file")
    parser.add_argument("--dev_set", type=str, default='/home/th/paddle/audio_segment/data/dataset/valid', required=False, help="The full path of dev_set_file")
    parser.add_argument("--test_set", type=str, default='/home/th/paddle/audio_segment/data/dataset/test', required=False, help="The full path of test_set_file")
    parser.add_argument("--chroma_dir", type=str, default='/home/th/paddle/audio_segment/data/features/chroma_features', required=False, help="The full path of chroma features file")
    parser.add_argument("--combine_dir", type=str, default='/home/th/paddle/audio_segment/data/features/combine_features', required=False, help="The full path of combine features file")
    parser.add_argument("--save_dir", default='/home/th/paddle/audio_segment/checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--max_steps', default=-1, type=int, help="If > 0, set total number of training steps to perform.")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--eval_step", default=100, type=int, help="Step interval for evaluation.")
    parser.add_argument("--log_step", default=20, type=int, help="Step interval for logging and printing loss.")
    parser.add_argument('--save_step', default=10000, type=int, help="Step interval for saving checkpoint.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--gpuids", type=str, default="-1", required=False, help="set gpu ids which use to perform")
    parser.add_argument("--do_test", type=bool, default=True, required=False, help="evaluate test set or not")
    # rnn parameters
    parser.add_argument("--input_size", type=int, default=140, required=False, help="set gpu ids which use to perform")
    parser.add_argument("--hidden_size", type=int, default=150, required=False, help="set gpu ids which use to perform")
    parser.add_argument("--num_layers", type=int, default=2, required=False, help="set gpu ids which use to perform")
    parser.add_argument("--rnn_style", type=str, default="lstm", required=False, help="set gpu ids which use to perform")
    # mel feature extract parameters 
    parser.add_argument("--frame_shift", type=int, default=10, required=False, help="frame shift per step / ms")
    parser.add_argument("--frame_length", type=int, default=25, required=False, help="frame length / ms")
    parser.add_argument("--n_mels", type=int, default=128, required=False, help="the number mels")
    parser.add_argument("--fmin", type=int, default=64, required=False, help="Hz")
    parser.add_argument("--fmax", type=int, default=8000, required=False, help="Hz")
    
    args = parser.parse_args()
    return args
# yapf: enable

def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, metric_acc, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric_acc.reset()
    losses = []
    total_num = 0
    start = time.time()
    true_labels = []
    pred_labels = []
    for batch in data_loader:
        input_features, labels = batch
        total_num += len(labels)
        logits = model(input_features=input_features)
        
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric_acc.compute(logits, labels)
        # acc
        metric_acc.update(correct)
        accu = metric_acc.accumulate()
        # precision
        preds = paddle.argmax(logits, axis=-1)
        true_labels += labels.tolist()
        pred_labels += preds.tolist()
    prec, rec, f1, true_sum = precision_recall_fscore_support(true_labels, pred_labels, beta=1)
    cost_times = time.time()-start
    logger.info("eval_loss: {:.4}, accuracy: {:.4}, eval_speed: {:.4} item/ms".format(
        np.mean(losses), accu, (cost_times / total_num) * 1000))
    for i in range(2):
        logger.info("label: {}, precision:{:.4}, recall:{:.4}, f1:{:.4}".format(id2label[i], prec[i], rec[i], f1[i]))
    model.train()
    metric_acc.reset()
    return accu


def train(args):
    if args.gpuids != "-1":
        args.device = args.device + ":" + args.gpuids
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    train_ds = load_dataset(read_data, args=args, audio_dir=args.train_set, lazy=False)
    dev_ds = load_dataset(read_data, args=args, audio_dir=args.dev_set, lazy=False)
    
    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.train_batch_size,
        batchify_fn=None,
        trans_fn=None)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.eval_batch_size,
        batchify_fn=None,
        trans_fn=None)

    model = AudioSegmentation(input_size=args.input_size, hidden_size=args.hidden_size, 
                                num_layers=args.num_layers, rnn_style=args.rnn_style)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs

    logger.info(f"The number of examples in train set: {len(train_ds)}")
    logger.info(f"The number of examples in dev set: {len(dev_ds)}")
    logger.info(f"All training steps: {num_training_steps}")

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = nn.loss.CrossEntropyLoss()
    metric_acc = paddle.metric.Accuracy()
    global_step = 0
    best_accuracy = 0.0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            
            input_features, labels = batch
            logits = model(input_features=input_features)
            correct = metric_acc.compute(logits, labels)
            metric_acc.update(correct)
            acc = metric_acc.accumulate()
            loss = criterion(logits, labels)
            global_step += 1
            if global_step % args.log_step == 0 and rank == 0:
                logger.info(
                    "global step %d, epoch: %d, loss: %.4f, train_acc: %.4f, speed: %.2f step/s"
                    % (global_step, epoch, loss, acc,
                       args.log_step / (time.time() - tic_train)))
                tic_train = time.time()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.eval_step == 0 and rank == 0:     
                accuracy = evaluate(model, criterion, metric_acc, dev_data_loader)
                if epoch > 1 and accuracy > best_accuracy:
                    save_dir = os.path.join(args.save_dir,
                                            "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir,
                                                   'model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    logger.info(f"****({best_accuracy}------->{accuracy})****")
                    logger.info(f"Saved the best model at {save_param_path}")
                    best_accuracy = accuracy

            if global_step == args.max_steps:
                return
    if args.do_test:
        logger.info("******Evaluating test set after training model******")
        state_dict = paddle.load(save_param_path)
        model.set_dict(state_dict)
        logger.info("Loaded parameters from %s" % save_param_path)
        test_ds = load_dataset(read_data, args=args, audio_dir=args.test_set, lazy=False)
        test_data_loader = create_dataloader(
                                        test_ds,
                                        mode='dev',
                                        batch_size=args.eval_batch_size,
                                        batchify_fn=None,
                                        trans_fn=None)
        evaluate(model, criterion, metric_acc, test_data_loader)
        

def run():
    args = getArgs()
    set_seed(args.seed)  
    train(args)


if __name__ == "__main__":
    run()