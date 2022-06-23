import argparse
import os
import random
import time
import logging

import numpy as np
import paddle
import paddle.nn as nn
from paddle.optimizer.lr import ExponentialDecay, LinearWarmup
# from sklearn.manifold import trustworthiness
# from paddle.metric import Accuracy
import paddle.nn.functional as F
# from sklearn.linear_model import LogisticRegression

from data import create_dataloader
from paddlenlp.datasets import load_dataset
from data_process.features_extract import read_data
from model import AudioSegmentation

from vauc_loss import VaucLoss

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# label2id = {
#     "no-music": 0,
#     "music": 1
# }

label2id = {
    "no-music": 0,
    "bg-music": 1,
    "fg-music": 2,
}

id2label = {}
for k, v in label2id.items():
    id2label[v] = k


def getArgs():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str, default='/home/th/paddle/audio_segment/data/dataset_2class/train', required=False, help="The full path of train_set_file")
    parser.add_argument("--dev_set", type=str, default='/home/th/paddle/audio_segment/data/dataset_2class/valid', required=False, help="The full path of dev_set_file")
    parser.add_argument("--test_set", type=str, default='/home/th/paddle/audio_segment/data/dataset_2class/test', required=False, help="The full path of test_set_file")
    # parser.add_argument("--chroma_dir", type=str, default='/home/th/paddle/audio_segment/data/features/chroma_features', required=False, help="The full path of chroma features file")
    parser.add_argument("--features_dir", type=str, default='/home/th/paddle/audio_segment/data/features/features_2c', required=False, help="The full path of combine features file")
    parser.add_argument("--save_dir", default='/home/th/paddle/audio_segment/checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--max_steps', default=-1, type=int, help="If > 0, set total number of training steps to perform.")
    parser.add_argument("--train_batch_size", default=48, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--eval_step", default=100, type=int, help="Step interval for evaluation.")
    parser.add_argument("--log_step", default=20, type=int, help="Step interval for logging and printing loss.")
    parser.add_argument('--save_step', default=10000, type=int, help="Step interval for saving checkpoint.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
    parser.add_argument("--init_from_ckpt", type=str, 
    default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--seed", type=int, default=100, help="Random seed for initialization.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--gpuids", type=str, default="0", required=False, help="set gpu ids which use to perform")
    parser.add_argument("--do_test", type=bool, default=True, required=False, help="evaluate test set or not")
    # rnn parameters set
    parser.add_argument("--input_size", type=int, default=64, required=False, help="set gpu ids which use to perform")
    parser.add_argument("--hidden_size", type=int, default=256, required=False, help="set gpu ids which use to perform")
    parser.add_argument("--num_layers", type=int, default=2, required=False, help="set gpu ids which use to perform")
    parser.add_argument("--rnn_style", choices=['lstm', 'gru'], default="gru", required=False, help="set gpu ids which use to perform")
    # mel feature extract parameters set 
    parser.add_argument("--frame_shift", type=int, default=10, required=False, help="frame shift per step / ms")
    parser.add_argument("--frame_length", type=int, default=25, required=False, help="frame length / ms")
    parser.add_argument("--n_mels", type=int, default=64, required=False, help="the number mels")
    parser.add_argument("--fmin", type=int, default=64, required=False, help="Hz")
    parser.add_argument("--fmax", type=int, default=8000, required=False, help="Hz")
    # loss parameters set 
    parser.add_argument("--loss", choices=['vauc-binary', 'vauc-ovo', "vauc-ovr"], 
                            default='vauc-binary', required=False, help="select which loss function to train")
    parser.add_argument("--delta", type=int, default=10, required=False, help="delta for controling the slope of the sigmoid in auc loss")

    parser.add_argument("--step", type=int, default=1, required=False, help="delta for controling the slope of the sigmoid in auc loss")
    args = parser.parse_args()
    return args
# yapf: enable
class metric:
    def __init__(self, num_class) -> None:

        self.num_class = num_class
        self.TPc = np.zeros(num_class)
        self.TNc = np.zeros(num_class) 
        self.FPc = np.zeros(num_class)
        self.FNc = np.zeros(num_class)

    def update(self, y_trues, y_preds, is_logits=False):
        if is_logits:
            y_preds = paddle.argmax(y_preds, axis=-1)
        y_trues = y_trues.flatten()
        y_preds = y_preds.flatten()
        for i in range(self.num_class):
            i_mask = y_trues == i
            trues_i = y_trues[i_mask]
            preds_i = y_preds[i_mask]
            if trues_i.shape[0] < 1:
                continue

            correct = (trues_i == preds_i).astype(paddle.int64)
            correct_n = paddle.sum(correct)
            self.TPc[i] += correct_n
            self.FNc[i] += (correct.shape[0] - correct_n)

            j_mask = paddle.logical_not(i_mask)
            # trues_j = y_trues[j_mask]
            preds_j = y_preds[j_mask]
            if preds_j.shape[0] < 1:
                continue
            neg_correct = (preds_j == i).astype(paddle.int64)
            neg_correct_n = paddle.sum(neg_correct)
            self.FPc[i] += neg_correct_n
            self.TNc[i] += (neg_correct.shape[0] - neg_correct_n)

    def pre_rec_f1(self):
        # class-wise Precision, Recall and F-measure
        pc = self.TPc / (self.TPc + self.FPc)  # [num_class]
        rc = self.TPc / (self.TPc + self.FNc)
        fc = 2 * pc * rc / (pc + rc)
        return pc, rc, fc

    def accuracy(self):
        TP = self.TPc.sum()
        FP = self.FPc.sum()
        TN = self.TNc.sum()
        FN = self.FNc.sum()
        acc = (TP + TN) / (TP + TN + FP + FN)
        return acc

    def reset(self):
        self.TPc = np.zeros(self.num_class)
        self.TNc = np.zeros(self.num_class) 
        self.FPc = np.zeros(self.num_class)
        self.FNc = np.zeros(self.num_class)

def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

# def compute_acc(preds, labels, is_logits=False):
#     # 样本准确率
#     if is_logits:
#         preds = paddle.argmax(preds, axis=-1)
#     correct = (preds == labels).astype(paddle.int64)
#     correct = (paddle.sum(correct, axis=-1) == 300).astype(paddle.int64)
#     n = paddle.sum(correct)
#     acc = n / correct.shape[0]
#     return acc.numpy().item(), n.numpy().item()
    
@paddle.no_grad()
def evaluate(model, criterion, metrics, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metrics.reset()
    
    losses = []
    total_num = 0
    start = time.time()
    for batch in data_loader:
        input_features, labels = batch
        total_num += len(labels)
        logits = model(input_features=input_features)
        # logits_re = logits.reshape((-1, len(id2label)))
        # labels_flat = labels.flatten()
        # vauc_loss, auc_loss = criterion(logits_re, labels_flat)
        # loss = 0.2 * vauc_loss + 0.8 * auc_loss
        # losses.append(loss.numpy())
        metrics.update(labels, logits, is_logits=True)
        
    prec, rec, f1 = metrics.pre_rec_f1()
    accu = metrics.accuracy()
    cost_times = time.time()-start
    logger.info("eval_loss: {:.4}, accuracy: {:.4}, eval_speed: {:.4} ms/item".format(
        0.0, accu, (cost_times / total_num) * 1000))
    for i in range(len(id2label)):
        logger.info("label: {}, precision:{:.4}, recall:{:.4}, f1:{:.4}".format(id2label[i], prec[i], rec[i], f1[i]))
    model.train()
    metrics.reset()
    return accu


def train(args):
    if args.gpuids != "-1":
        args.device = args.device + ":" + args.gpuids
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    train_ds = load_dataset(read_data, args=args, data_dir=args.train_set, label2id=label2id, k=None, lazy=False)
    dev_ds = load_dataset(read_data, args=args, data_dir=args.dev_set, label2id=label2id, k=None, lazy=False)
    
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
                                num_layers=args.num_layers, num_class=len(id2label), rnn_style=args.rnn_style)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        logger.info(f"Initializing model from {args.init_from_ckpt}")
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        
    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs

    logger.info(f"The number of examples in train set: {len(train_ds)}")
    logger.info(f"The number of examples in dev set: {len(dev_ds)}")
    logger.info(f"All training steps: {num_training_steps}")

    lr_scheduler = ExponentialDecay(learning_rate=args.learning_rate, gamma=0.895)
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        parameters=model.parameters())
    
    criterion = VaucLoss(mutil_class=args.loss.split("-")[-1], delta=args.delta)
    
    logger.info(f"****** Training model on {args.loss} loss function ******")
    metrics = metric(num_class=len(id2label))
    
    global_step = 0
    best_accuracy = 0.0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            
            input_features, labels = batch
            logits = model(input_features=input_features)
            logits_re = logits.reshape((-1, len(id2label)))
            labels_flat = labels.flatten()
            metrics.update(labels, logits, is_logits=True)
            acc = metrics.accuracy() 
            vauc_loss, auc_loss = criterion(logits_re, labels_flat)
            loss = vauc_loss
            global_step += 1
            if global_step % args.log_step == 0 and rank == 0:
                logger.info(
                    "global step %d, epoch: %d, loss: %.4f, auc_loss: %.4f, vauc_loss: %f, train_acc: %.4f, speed: %.2f step/s"
                    % (global_step, epoch, loss, auc_loss, vauc_loss * 1000000, acc,
                       args.log_step / (time.time() - tic_train)))
                tic_train = time.time()

            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch=epoch)
            optimizer.clear_grad()

            if global_step % args.eval_step == 0 and rank == 0:     
                accuracy = evaluate(model, criterion, metrics, dev_data_loader)
                if epoch > 1 and accuracy > best_accuracy:
                    save_dir = os.path.join(args.save_dir + "/" + str(args.step) + "/" + args.loss,
                                            "vauc_model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    logger.info(f"******({best_accuracy}------->{accuracy})******")
                    logger.info(f"Saved the best model at {save_param_path}")
                    best_accuracy = accuracy

            if global_step == args.max_steps:
                return
        
    if args.do_test:
        logger.info("****** Evaluating test set after training model ******")
        # save_param_path = "/home/th/paddle/audio_segment/checkpoint/ce/model_8800/model_state.pdparams"
        # save_param_path = "/home/th/paddle/audio_segment/checkpoint/auc-ovo/model_8300/model_state.pdparams"
        state_dict = paddle.load(save_param_path)
        model.set_dict(state_dict)
        logger.info("Loaded parameters from %s" % save_param_path)
        test_ds = load_dataset(read_data, args=args, data_dir=args.test_set, label2id=label2id, lazy=False)
        
        test_data_loader = create_dataloader(
                                        test_ds,
                                        mode='dev',
                                        batch_size=args.eval_batch_size,
                                        batchify_fn=None,
                                        trans_fn=None)
        logger.info(f"The number of examples in test set: {len(test_ds)}")
        evaluate(model, criterion, metrics, test_data_loader)



def run():
    args = getArgs()
    set_seed(args.seed)  
    train(args)


if __name__ == "__main__":
    run()