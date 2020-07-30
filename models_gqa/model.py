from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from apex import amp

from . import ops as ops
from .config import cfg
from .lcgn import LCGN, SemanLCGN
from .input_unit import Encoder
from .output_unit import Classifier
from .optimization import *

class SingleHop(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = ops.Linear(cfg.ENC_DIM, cfg.CTX_DIM)
        self.inter2att = ops.Linear(cfg.CTX_DIM, 1)

    def forward(self, kb, vecQuestions, imagesObjectNum):
        proj_q = self.proj_q(vecQuestions)
        interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
        raw_att = self.inter2att(interactions).squeeze(-1)# 128 * 49
        raw_att = ops.apply_mask1d(raw_att, imagesObjectNum)
        att = F.softmax(raw_att, dim=-1)

        x_att = torch.bmm(att[:, None, :], kb).squeeze(1)
        return x_att


class LCGNnet(nn.Module):
    def __init__(self, num_vocab, num_choices):
        super().__init__()
        if cfg.INIT_WRD_EMB_FROM_FILE:
            embeddingsInit = np.load(cfg.WRD_EMB_INIT_FILE) # 2956 * 300
            assert embeddingsInit.shape == (num_vocab-1, cfg.WRD_EMB_DIM)
        else:
            embeddingsInit = np.random.randn(num_vocab-1, cfg.WRD_EMB_DIM)
        self.num_vocab = num_vocab # 2957
        self.num_choices = num_choices # 1845
        self.encoder = Encoder(embeddingsInit)
        self.lcgn = LCGN()
        self.sema_lcgn = SemanLCGN()
        self.single_hop = SingleHop()
        self.classifier = Classifier(num_choices)
        self.seman_encoder = ops.Linear(cfg.WRD_EMB_DIM, cfg.CMD_DIM)
        self.conv1d = nn.Conv1d(cfg.CMD_DIM, out_channels=cfg.CMD_DIM, kernel_size=4)

    def forward(self, batch):
        #batchSize = len(batch['image_feat_batch'])
        questionIndices = batch[0]
        questionLengths = batch[1]
        semanIndices = batch[2]
        semanLengths = batch[3]
        answerIndices = batch[4]
        images = batch[5]
        imagesObjectNum = batch[6]
        batchSize = images.size(0)
        # LSTM
        questionCntxWords, vecQuestions, semanCnt = self.encoder(
            questionIndices, questionLengths, # 128 * 30 * 512 128 * 512
            semanIndices, semanLengths) 

        # LCGN
        x_out = self.lcgn(
            images=images, q_encoding=vecQuestions,
            lstm_outputs=questionCntxWords, batch_size=batchSize,
            q_length=questionLengths, entity_num=imagesObjectNum)

        # semanCnt = semanCnt.permute(1, 0, 2)
        # semanCnt = semanCnt[:, 0, :]
        semanCnt = self.seman_encoder(semanCnt)

        x_out_seman = self.sema_lcgn(
            images=images, seman_outputs=semanCnt,
            batch_size=batchSize, entity_num=imagesObjectNum)

        x_out = self.tensor_inter_graph_propagation(x_out, x_out_seman)
        # Single-Hop
        x_att = self.single_hop(x_out, vecQuestions, imagesObjectNum)
        logits = self.classifier(x_att, vecQuestions) # 128 * 1845

        predictions, num_correct = self.add_pred_op(logits, answerIndices)
        loss = self.add_answer_loss_op(logits, answerIndices)

        return {"predictions": predictions,
                "batch_size": int(batchSize),
                "num_correct": int(num_correct),
                "loss": loss,
                "accuracy": float(num_correct * 1. / batchSize)}

    def tensor_inter_graph_propagation(self, x_out_1, x_out_2):
        bsz, imageNum, dModel= x_out_1.size(0), x_out_1.size(1), x_out_1.size(2)
        x_sum_1 = torch.sum(x_out_1, dim=1)
        x_sum_2 = torch.sum(x_out_2, dim=1)

        x_expand_1 = x_sum_1.repeat(1, 2)
        x_expand_2 = x_sum_2.repeat(1, 2)

        x_sum = torch.cat([x_expand_1, x_expand_2], -1)
        x_sum = x_sum.unsqueeze(1)
        x_sum = x_sum.repeat(1, imageNum, 1)

        x_union = torch.cat([x_out_1, x_out_2], dim=-1)
        x_union_expand = x_union.repeat(1, 1, 2)

        x_kr = torch.mul(x_union_expand, x_sum)
        x_kr = x_kr.view(bsz * imageNum, 4, dModel)
        x_kr = x_kr.permute(0, 2, 1)

        x_out = self.conv1d(x_kr)
        x_out = x_out.squeeze(-1)
        x_out = x_out.view(bsz, imageNum, dModel)
        
        return x_out
    
    def add_pred_op(self, logits, answers):
        if cfg.MASK_PADUNK_IN_LOGITS:
            logits = logits.clone()
            logits[..., :2] += -1e30  # mask <pad> and <unk>

        preds = torch.argmax(logits, dim=-1).detach() # 128
        corrects = (preds == answers)
        correctNum = torch.sum(corrects).item()
        preds = preds.cpu()#.numpy()

        return preds, correctNum

    def add_answer_loss_op(self, logits, answers):
        if cfg.TRAIN.LOSS_TYPE == "softmax":
            loss = F.cross_entropy(logits, answers)
        elif cfg.TRAIN.LOSS_TYPE == "sigmoid":
            answerDist = F.one_hot(answers, self.num_choices).float() # 128 * 1845
            loss = F.binary_cross_entropy_with_logits(
                logits, answerDist) * self.num_choices
        else:
            raise Exception("non-identified loss")
        return loss


class LCGNwrapper():
    def __init__(self, num_vocab, num_choices, cfg=None, rank=-1, gpu=0):

        self.no_decay = ['bias']

        torch.cuda.set_device(gpu)
        self.model = LCGNnet(num_vocab, num_choices).cuda(gpu)

        self.trainable_params = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in self.no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in self.no_decay)],
                "weight_decay": 0.0
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
                self.trainable_params, lr=cfg.TRAIN.SOLVER.LR)
        #self.optimizer = AdamW(self.trainable_params, lr=cfg.TRAIN.SOLVER.LR, eps=cfg.adam_epsilon)
        total_step = int(943000 / cfg.n_gpus // cfg.TRAIN.BATCH_SIZE + 1) * cfg.TRAIN.MAX_EPOCH
        self.scheduler = get_linear_schedule_with_warmup(
                                                self.optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_step)
        if cfg.fp16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=cfg.fp16_opt_level)

        if cfg.n_gpus > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                                device_ids=[gpu], output_device=gpu, find_unused_parameters=True)


        self.lr = cfg.TRAIN.SOLVER.LR
        self.fp16 = cfg.fp16
        self.fp16_opt_level = cfg.fp16_opt_level

        if cfg.USE_EMA:
            self.ema_param_dict = {
                name: p for name, p in self.model.named_parameters()
                if p.requires_grad}
            self.ema = ops.ExponentialMovingAverage(
                self.ema_param_dict, decay=cfg.EMA_DECAY_RATE)
            self.using_ema_params = False

    def train(self, training=True):
        self.model.train(training)
        if training:
            self.set_params_from_original()
        else:
            self.set_params_from_ema()

    def eval(self):
        self.train(False)

    def state_dict(self):
        # Generate state dict in training mode
        current_mode = self.model.training
        self.train(True)

        assert (not cfg.USE_EMA) or (not self.using_ema_params)
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict() if cfg.USE_EMA else None
        }

        # restore original mode
        self.train(current_mode)

    def load_state_dict(self, state_dict):
        # Load parameters in training mode
        current_mode = self.model.training
        self.train(True)

        assert (not cfg.USE_EMA) or (not self.using_ema_params)
        self.model.load_state_dict(state_dict['model'])

        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            print('Optimizer does not exist in checkpoint! '
                  'Loaded only model parameters.')

        if cfg.USE_EMA:
            if 'ema' in state_dict and state_dict['ema'] is not None:
                self.ema.load_state_dict(state_dict['ema'])
            else:
                print('cfg.USE_EMA is True, but EMA does not exist in '
                      'checkpoint! Using model params to initialize EMA.')
                self.ema.load_state_dict(
                    {k: p.data for k, p in self.ema_param_dict.items()})

        # restore original mode
        self.train(current_mode)

    def set_params_from_ema(self):
        if (not cfg.USE_EMA) or self.using_ema_params:
            return

        self.original_state_dict = deepcopy(self.model.state_dict())
        self.ema.set_params_from_ema(self.ema_param_dict)
        self.using_ema_params = True

    def set_params_from_original(self):
        if (not cfg.USE_EMA) or (not self.using_ema_params):
            return

        self.model.load_state_dict(self.original_state_dict)
        self.using_ema_params = False

    def run_batch(self, batch, train, lr=None):
        assert train == self.model.training
        assert (not train) or (lr is not None), 'lr must be set for training'

        if train:
            if lr != self.lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
            self.optimizer.zero_grad()
            batch_res = self.model.forward(batch)
            loss = batch_res['loss']
            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if cfg.TRAIN.CLIP_GRADIENTS:
                if self.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), cfg.TRAIN.GRAD_MAX_NORM)
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.TRAIN.GRAD_MAX_NORM)
            self.optimizer.step()
            self.scheduler.step()
            batch_res['lr'] = self.scheduler.get_lr()[0]
            if cfg.USE_EMA:
                self.ema.step(self.ema_param_dict)
        else:
            with torch.no_grad():
                batch_res = self.model.forward(batch)

        return batch_res
