import os
import json
import torch
import sys
import time
import random
import numpy as np

from tqdm import tqdm, trange
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

sys.path.append('../')
from models_gqa.model import LCGNwrapper
from models_gqa.config import build_cfg_from_argparse
from util.gqa_train.data_reader import DataReader
#from util.gqa_train.data_reader import gqa_convert_examples_to_features
# Load config
# cmd = '--cfg /home/xdjf/lcgn-pytorch/exp_gqa/cfgs/lcgn_spatial.yaml train True'.split()
# sys.argv.extend(cmd)

# Start session
#os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS
# if len(cfg.GPUS.split(',')) > 1:
#     print('PyTorch implementation currently only supports single GPU')
import wandb


def load_train_data(cfg, rank, gpu, max_num=0, num_replicas=1):
    imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_VQA
    scene_graph_file = cfg.SCENE_GRAPH_FILE % \
        cfg.TRAIN.SPLIT_VQA.replace('_balanced', '').replace('_all', '')
    #a = gqa_convert_examples_to_features(imdb_file, scene_graph_file, cfg)

    data_reader = DataReader(
        imdb_file, rank, gpu, num_replicas, shuffle=True, max_num=max_num,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        vocab_question_file=cfg.VOCAB_QUESTION_FILE,
        T_encoder=cfg.T_ENCODER,
        N_encoder=cfg.N_ENCODER,
        vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
        feature_type=cfg.FEAT_TYPE,
        spatial_feature_dir=cfg.SPATIAL_FEATURE_DIR,
        objects_feature_dir=cfg.OBJECTS_FEATURE_DIR,
        objects_max_num=cfg.W_FEAT,
        scene_graph_file=scene_graph_file,
        vocab_name_file=cfg.VOCAB_NAME_FILE,
        vocab_attr_file=cfg.VOCAB_ATTR_FILE,
        add_pos_enc=cfg.ADD_POS_ENC,
        pos_enc_dim=cfg.PE_DIM, 
        pos_enc_scale=cfg.PE_SCALE)
    num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
    num_choices = data_reader.batch_loader.answer_dict.num_vocab
    return data_reader, num_vocab, num_choices


def batch_to_data(batch):
    questionIndices = torch.from_numpy(
            batch['input_seq_batch'].astype(np.int64)).cuda() # 128 * 30
    questionLengths = torch.from_numpy(
        batch['seq_length_batch'].astype(np.int64)).cuda() # 128
    semanIndices = torch.from_numpy(
        batch['input_seman_batch'].astype(np.int64)).cuda() # 128 * 30
    semanLengths = torch.from_numpy(
        batch['seman_length_batch'].astype(np.int64)).cuda() # 128
    answerIndices = torch.from_numpy(
        batch['answer_label_batch'].astype(np.int64)).cuda() # 128
    images = torch.from_numpy(
        batch['image_feat_batch'].astype(np.float32)).cuda() # 128 * 49 * 2112
    imagesObjectNum = torch.from_numpy(
        np.sum(batch['image_valid_batch'].astype(np.int64), axis=1)).cuda() # 128
    return (questionIndices, questionLengths, semanIndices, semanLengths, answerIndices, images, imagesObjectNum)

def run_train_on_data(model, data_reader_train, cfg, rank, gpu, run_eval=False,
                      data_reader_eval=None):
    model.train()

    global_step = 1
    lr = cfg.TRAIN.SOLVER.LR
    correct, total, loss_sum, batch_num = 0, 0, 0., 0
    tr_loss, logging_loss = 0.0, 0.0

    # if rank in [-1, 0]:
    #     tb_writer = SummaryWriter()
    
    for batch, n_sample, e in data_reader_train.batches(one_pass=False):
        n_epoch = cfg.TRAIN.START_EPOCH + e
        if n_sample == 0 and n_epoch > cfg.TRAIN.START_EPOCH and rank in [-1, 0]:
            print('')
            # save snapshot
            snapshot_file = cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, n_epoch)
            torch.save(model.state_dict(), snapshot_file)
            # run evaluation
            if run_eval:
                batch_eval = run_eval_on_data(cfg, model, data_reader_eval)
                #tb_writer.add_scalar("eval_loss", batch_eval['loss'], global_step)
                model.train()
            if cfg.DEBUG == False:
                wandb.log({"eval_loss": batch_eval['loss'], "eval_correct": batch_eval['accuracy']})
            # clear stats
            correct, total, loss_sum, batch_num = 0, 0, 0., 0
        if n_epoch >= cfg.TRAIN.MAX_EPOCH:
            break

        batch_list = batch_to_data(batch)
        # if first and rank in [-1, 0]:
        #     tb_writer.add_graph(model.model, (batch_list, ))
        #     first = False
        
        batch_res = model.run_batch(batch_list, train=True, lr=lr)
        correct += batch_res['num_correct']
        total += batch_res['batch_size']
        loss_sum += batch_res['loss'].item()
        tr_loss += loss_sum
        batch_num += 1
        global_step += 1
        lr = batch_res['lr']
        
        if rank in [-1, 0] and cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0 and cfg.DEBUG == False:
            wandb.log({"lr": batch_res['lr'], "train_loss": loss_sum/batch_num, "train_correct": correct/total})
            # tb_writer.add_scalar("lr", batch_res['lr'], global_step)
            # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / cfg.logging_steps, global_step)

        if rank in [-1, 0]:
            print('\rTrain E %d S %d: avgL=%.4f, avgA=%.4f, lr=%.1e' % (
                    n_epoch+1, total, loss_sum/batch_num, correct/total, lr),
                end='')

    # if rank in [-1, 0]:
    #     tb_writer.close()


def load_eval_data(cfg, rank, gpu, max_num=0):
    imdb_file = cfg.IMDB_FILE % cfg.TEST.SPLIT_VQA
    scene_graph_file = cfg.SCENE_GRAPH_FILE % \
        cfg.TEST.SPLIT_VQA.replace('_balanced', '').replace('_all', '')
    data_reader = DataReader(
        imdb_file, rank, gpu, 1, shuffle=False, max_num=max_num,
        batch_size=cfg.TEST.BATCH_SIZE,
        vocab_question_file=cfg.VOCAB_QUESTION_FILE,
        T_encoder=cfg.T_ENCODER,
        N_encoder=cfg.N_ENCODER,
        vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
        feature_type=cfg.FEAT_TYPE,
        spatial_feature_dir=cfg.SPATIAL_FEATURE_DIR,
        objects_feature_dir=cfg.OBJECTS_FEATURE_DIR,
        objects_max_num=cfg.W_FEAT,
        scene_graph_file=scene_graph_file,
        vocab_name_file=cfg.VOCAB_NAME_FILE,
        vocab_attr_file=cfg.VOCAB_ATTR_FILE,
        add_pos_enc=cfg.ADD_POS_ENC,
        pos_enc_dim=cfg.PE_DIM, pos_enc_scale=cfg.PE_SCALE)
    num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
    num_choices = data_reader.batch_loader.answer_dict.num_vocab
    return data_reader, num_vocab, num_choices


def run_eval_on_data(cfg, model, data_reader_eval, pred=False):
    model.eval()
    predictions = []
    answer_tokens = data_reader_eval.batch_loader.answer_dict.word_list
    correct, total, loss_sum, batch_num = 0, 0, 0., 0
    for batch, _, _ in data_reader_eval.batches(one_pass=True):
        batch_list = batch_to_data(batch)
        batch_res = model.run_batch(batch_list, train=False)
        if pred:
            predictions.extend([
                {'questionId': q, 'prediction': answer_tokens[p]}
                for q, p in zip(batch['qid_list'], batch_res['predictions'])])
        correct += batch_res['num_correct']
        total += batch_res['batch_size']
        loss_sum += batch_res['loss'].item()
        batch_num += 1
        print('\rEval S %d: avgL=%.4f, avgA=%.4f' % (
            total, loss_sum/batch_num, correct/total), end='')
    print('')
    eval_res = {
        'correct': correct,
        'total': total,
        'accuracy': correct/total,
        'loss': loss_sum/batch_num,
        'predictions': predictions}
    return eval_res


def dump_prediction_to_file(cfg, predictions, res_dir):
    pred_file = os.path.join(res_dir, 'pred_%s_%04d_%s.json' % (
        cfg.EXP_NAME, cfg.TEST.EPOCH, cfg.TEST.SPLIT_VQA))
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print('predictions written to %s' % pred_file)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(gpu, cfg):

    rank = -1
    if gpu != -1:
        rank = cfg.nr * cfg.n_gpus + gpu	  
        dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=cfg.world_size,                              
    	rank=rank                                               
        )  
    if rank in [-1, 0, 1]:
        gpu = 0
    elif rank in [2, 3]:
        gpu = 1
    
    set_seed(cfg)

    print(f'rank: {rank} pid: {os.getpid()} is running...')
    num_replicas = cfg.world_size if rank != -1 else 1
    data_reader_train, num_vocab, num_choices = load_train_data(cfg, rank, gpu, num_replicas=num_replicas)
    data_reader_eval, _, _ = load_eval_data(cfg, rank, gpu, max_num=cfg.TRAIN.EVAL_MAX_NUM)
    # Load model

    model = LCGNwrapper(num_vocab, num_choices, cfg=cfg, rank=rank, gpu=gpu)
    # Save snapshot
    if rank in [-1, 0]:
        if cfg.DEBUG == False:
            name = time.strftime('%Y%m%d-%H%M%S')
            wandb.init(project="gtp", notes="graph tensor propa", name=name)
            wandb.watch(model.model, log="all")
            wandb.config.update(cfg)
        snapshot_dir = os.path.dirname(cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, 0))
        os.makedirs(snapshot_dir, exist_ok=True)
        with open(os.path.join(snapshot_dir, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
    if cfg.TRAIN.START_EPOCH > 0 and rank in [-1, 0]:
        print('resuming from epoch %d' % cfg.TRAIN.START_EPOCH)
        model.load_state_dict(torch.load(
            cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, cfg.TRAIN.START_EPOCH)))

    if rank in [-1, 0]:
        print('%s - train for %d epochs' % (cfg.EXP_NAME, cfg.TRAIN.MAX_EPOCH))
    run_train_on_data(
        model, data_reader_train, cfg, rank, gpu, run_eval=cfg.TRAIN.RUN_EVAL,
        data_reader_eval=data_reader_eval)
    if rank in [-1, 0]:
        print('%s - train (done)' % cfg.EXP_NAME)


def test(cfg):
    data_reader_eval, num_vocab, num_choices = load_eval_data(cfg, -1, 0)

    # Load model
    model = LCGNwrapper(num_vocab, num_choices, cfg)

    # Load test snapshot
    snapshot_file = cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, cfg.TEST.EPOCH)
    model.load_state_dict(torch.load(snapshot_file))

    res_dir = cfg.TEST.RESULT_DIR % (cfg.EXP_NAME, cfg.TEST.EPOCH)
    vis_dir = os.path.join(
        res_dir, '%s_%s' % (cfg.TEST.VIS_DIR_PREFIX, cfg.TEST.SPLIT_VQA))
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    pred = cfg.TEST.DUMP_PRED
    if not pred:
        print('NOT writing predictions (set TEST.DUMP_PRED True to write)')

    print('%s - test epoch %d' % (cfg.EXP_NAME, cfg.TEST.EPOCH))
    eval_res = run_eval_on_data(cfg, model, data_reader_eval, pred=pred)
    print('%s - test epoch %d: accuracy = %.4f' % (
        cfg.EXP_NAME, cfg.TEST.EPOCH, eval_res['accuracy']))

    # write results
    if pred:
        dump_prediction_to_file(cfg, eval_res['predictions'], res_dir)
    eval_res.pop('predictions')
    res_file = os.path.join(res_dir, 'res_%s_%04d_%s.json' % (
        cfg.EXP_NAME, cfg.TEST.EPOCH, cfg.TEST.SPLIT_VQA))
    with open(res_file, 'w') as f:
        json.dump(eval_res, f)


if __name__ == '__main__':

    cfg = build_cfg_from_argparse()
    start = time.time()
    print(f'pid: {os.getpid()} is running...')
    if cfg.train:
        if cfg.n_gpus > 1:
            os.environ['MASTER_ADDR'] = '127.0.0.1' 
            os.environ['MASTER_PORT'] = '12801'     
            cfg.world_size = cfg.n_gpus * cfg.nodes
            mp.spawn(train, nprocs=cfg.n_gpus, args=(cfg,))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS
            train(-1, cfg)
        end_ = time.time()
        if cfg.DEBUG == False:
            wandb.log({"training time": int((end_ - start) / 60)})
        print(f'time has cost : {end_ - start}')
    else:
        test(cfg)
