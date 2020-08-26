import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import ops as ops
from .config import cfg


class GatedTrans(nn.Module):
    """docstring for GatedTrans"""
    def __init__(self, in_dim, out_dim):
        super(GatedTrans, self).__init__()
        
        self.embed_y = nn.Sequential(
            ops.Linear(
                in_dim,
                out_dim
            ),
            nn.Tanh()
        )
        self.embed_g = nn.Sequential(
            ops.Linear(
                in_dim,
                out_dim
            ),
            nn.Sigmoid()
        )

    def forward(self, x_in):
        x_y = self.embed_y(x_in)
        x_g = self.embed_g(x_in)
        x_out = x_y * x_g

        return x_out


class Self_Att(nn.Module):
    """Self attention module of questions."""
    def __init__(self):
        super(Self_Att, self).__init__()

        self.embed = nn.Sequential(
            nn.Dropout(p=cfg["dropout_fc"]),
            GatedTrans(
                cfg["WRD_EMB_DIM"],
                cfg["CMD_DIM"]
            ),
        )        
        self.att = nn.Sequential(
            nn.Dropout(p=cfg["dropout_fc"]),
            nn.Linear(
                cfg["CMD_DIM"],
                1
            )
        )
        self.word_e = ops.Linear(cfg["WRD_EMB_DIM"], cfg["CMD_DIM"])
        self.softmax = nn.Softmax(dim=-1)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight.data)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias.data, 0)

    def forward(self, word, word_encoded, word_not_pad):
        # ques_word shape: (batch_size, num_rounds, quen_len_max, word_embed_dim)
        # ques_embed shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size * 2)
        # ques_not_pad shape: (batch_size, num_rounds, quen_len_max)
        # output: img_att (batch_size, num_rounds, embed_dim)
        batch_size = word.size(0)
        len_max = word.size(1)

        word_embed = self.embed(word_encoded) # shape: (batch_size, num_rounds, quen_len_max, embed_dim)
        word_norm = F.normalize(word_embed, p=2, dim=-1) # shape: (batch_size, num_rounds, quen_len_max, embed_dim) 
        
        att = self.att(word_norm).squeeze(-1) # shape: (batch_size, num_rounds, quen_len_max)
        word = self.word_e(word)
        # ignore <pad> word
        att = self.softmax(att)
        att = att * word_not_pad # shape: (batch_size, num_rounds, quen_len_max)
        att = att / torch.sum(att, dim=-1, keepdim=True) # shape: (batch_size, num_rounds, quen_len_max)
        feat = torch.sum(att.unsqueeze(-1) * word, dim=-2) # shape: (batch_size, num_rounds, rnn_dim)
        
        return feat, att


class LCGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.build_loc_ctx_init()
        self.build_extract_textual_command()
        self.build_extract_seman_command()
        self.build_propagate_message()

    def build_loc_ctx_init(self):
        assert cfg.STEM_LINEAR != cfg.STEM_CNN
        if cfg.STEM_LINEAR:
            self.initKB = ops.Linear(cfg.D_FEAT, cfg.CTX_DIM)
            self.x_loc_drop = nn.Dropout(1 - cfg.stemDropout)
        elif cfg.STEM_CNN:
            self.cnn = nn.Sequential(
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.D_FEAT, cfg.STEM_CNN_DIM, (3, 3), padding=1),
                nn.ELU(),
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.STEM_CNN_DIM, cfg.CTX_DIM, (3, 3), padding=1),
                nn.ELU())

        self.initMem = nn.Parameter(torch.randn(1, 1, cfg.CTX_DIM))

    def build_extract_textual_command(self):
        self.qInput = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
        for t in range(cfg.MSG_ITER_NUM):
            qInput_layer2 = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
            setattr(self, "qInput%d" % t, qInput_layer2)
        self.cmd_inter2logits = ops.Linear(cfg.CMD_DIM, 1)
    
    def build_extract_seman_command(self):
        for t in range(cfg.MSG_ITER_NUM):
            seman_layer = Self_Att()
            setattr(self, "Seman%d" % t, seman_layer)

    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - cfg.readDropout)
        self.project_x_loc = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.project_x_ctx = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.queries = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.keys = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.vals = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.proj_keys = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.proj_vals = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.mem_update = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)
        self.combine_kb = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)

        self.read_drop_seman = nn.Dropout(1 - cfg.readDropout)
        self.project_x_loc_seman = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.project_x_ctx_seman = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.queries_seman = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        # self.keys_seman = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        # self.vals_seman = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.proj_keys_seman = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.proj_vals_seman = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.mem_update_seman = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)

        self.keys_name = ops.Linear(cfg.NAME_DIM, cfg.CTX_DIM)
        self.vals_name = ops.Linear(cfg.NAME_DIM, cfg.CTX_DIM)
        #self.combine_kb_seman = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)

        self.layernorm_q = nn.LayerNorm(cfg.CTX_DIM)
        self.layernorm_s = nn.LayerNorm(cfg.CTX_DIM)
        self.conv1d_q = nn.Conv1d(4, out_channels=1, kernel_size=1)
        self.conv1d_s = nn.Conv1d(4, out_channels=1, kernel_size=1)


    def forward(self, images, q_encoding, lstm_outputs, word_seman, encode_seman, semanIndices, batch_size, q_length,
                entity_num, name_embed, nameLengths, norm=True):
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(images)
        seman_not_pad = (semanIndices != 0).float()
        for t in range(cfg.MSG_ITER_NUM):
            if t == 0:
                x_ctx_q = self.run_message_passing_iter(
                    q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
                    x_ctx_var_drop, entity_num, t)
                x_ctx_s = self.run_message_passing_seman_iter(
                    word_seman, encode_seman, seman_not_pad, x_loc, x_ctx,
                    x_ctx_var_drop, entity_num, name_embed, nameLengths, t)
            else:
                x_ctx_q = self.run_message_passing_iter(
                    q_encoding, lstm_outputs, q_length, x_loc, x_ctx_q,
                    x_ctx_var_drop, entity_num, t)
                x_ctx_s = self.run_message_passing_seman_iter(
                    word_seman, encode_seman, seman_not_pad, x_loc, x_ctx_s,
                    x_ctx_var_drop, entity_num, name_embed, nameLengths, t)
            x_ctx_q, x_ctx_s = self.tensor_inter_graph_propagation(x_ctx_q, x_ctx_s, entity_num)
            if norm:
                x_ctx_q = self.layernorm_q(x_ctx_q)
                x_ctx_s = self.layernorm_s(x_ctx_s)
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx_q, x_ctx_s], dim=-1))
        return x_out

    def extract_textual_command(self, q_encoding, lstm_outputs, q_length, t):
        qInput_layer2 = getattr(self, "qInput%d" % t)
        act_fun = ops.activations[cfg.CMD_INPUT_ACT]
        q_cmd = qInput_layer2(act_fun(self.qInput(q_encoding))) # 128 * 512
        raw_att = self.cmd_inter2logits(
            q_cmd[:, None, :] * lstm_outputs).squeeze(-1) # 128 * 1 * 512 128 * 30 * 512 -> 128 * 30
        raw_att = ops.apply_mask1d(raw_att, q_length)
        att = F.softmax(raw_att, dim=-1) # 128 * 30
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1) # 128 * 512
        return cmd

    def extract_seman_command(self, word_seman, encode_seman, seman_not_pad, t):
        seman_layer = getattr(self, "Seman%d" % t)
        seman_cmd, att = seman_layer(word_seman, encode_seman, seman_not_pad)
        return seman_cmd

    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint) # 128 * 49 * 512
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :] # 128 * 49 * 512
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :] # 128 * 49 * 512
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM)) # 128 * 49 * 49
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new
    
    def propagate_seman_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, name_embed, nameLengths, entity_num):
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc_seman(self.read_drop_seman(x_loc))
        proj_x_ctx = self.project_x_ctx_seman(self.read_drop_seman(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries_seman(x_joint) # 128 * 49 * 512
        keys = self.keys_name(name_embed) * self.proj_keys_seman(cmd)[:, None, :] # 128 * 49 * 512
        vals = self.vals_name(name_embed) * self.proj_vals_seman(cmd)[:, None, :] # 128 * 49 * 512
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM)) # 128 * 49 * 49
        edge_score = ops.apply_mask2d_name(edge_score, entity_num, nameLengths)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update_seman(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new

    def run_message_passing_iter(
            self, q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
            x_ctx_var_drop, entity_num, t):
        cmd = self.extract_textual_command(
                q_encoding, lstm_outputs, q_length, t)
        x_ctx = self.propagate_message(
            cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num)
        return x_ctx
    
    def run_message_passing_seman_iter(
            self, word_seman, encode_seman, seman_not_pad, x_loc, x_ctx,
            x_ctx_var_drop, entity_num, name_embed, nameLengths, t):
        cmd = self.extract_seman_command(
                word_seman, encode_seman, seman_not_pad, t)
        x_ctx = self.propagate_seman_message(
            cmd, x_loc, x_ctx, x_ctx_var_drop, name_embed, nameLengths, entity_num)
        return x_ctx

    def loc_ctx_init(self, images):
        if cfg.STEM_NORMALIZE:
            images = F.normalize(images, dim=-1)
        if cfg.STEM_LINEAR:
            x_loc = self.initKB(images) # 128 * 49 * 512
            x_loc = self.x_loc_drop(x_loc)
        elif cfg.STEM_CNN:
            images = torch.transpose(images, 1, 2)  # N(HW)C => NC(HW)
            x_loc = images.view(-1, cfg.D_FEAT, cfg.H_FEAT, cfg.W_FEAT)
            x_loc = self.cnn(x_loc)
            x_loc = x_loc.view(-1, cfg.CTX_DIM, cfg.H_FEAT * cfg.W_FEAT)
            x_loc = torch.transpose(x_loc, 1, 2)  # NC(HW) => N(HW)C
        if cfg.STEM_RENORMALIZE:
            x_loc = F.normalize(x_loc, dim=-1)

        x_ctx = self.initMem.expand(x_loc.size()) # 128 * 49 * 512
        x_ctx_var_drop = ops.generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(cfg.memoryDropout if self.training else 1.))
        return x_loc, x_ctx, x_ctx_var_drop



    def tensor_inter_graph_propagation(self, x_out_1, x_out_2, entity_num):
        bsz, imageNum, dModel = x_out_1.size(0), x_out_1.size(1), x_out_1.size(2)

        att_sum = torch.ones((bsz, imageNum), dtype=x_out_1.dtype, device=x_out_1.device)
        mask_sum = ops.apply_mask1d(att_sum, entity_num)

        x_sum_1 = torch.bmm(mask_sum[:, None, :], x_out_1).squeeze(1)
        x_sum_2 = torch.bmm(mask_sum[:, None, :], x_out_2).squeeze(1)
        # x_sum_1 = torch.sum(x_out_1, dim=1)
        # x_sum_2 = torch.sum(x_out_2, dim=1)

        x_expand_1 = x_sum_1.repeat(1, 2)
        x_expand_2 = x_sum_2.repeat(1, 2)

        x_sum = torch.cat([x_expand_1, x_expand_2], -1)
        x_sum = x_sum.unsqueeze(1)
        x_sum = x_sum.repeat(1, imageNum, 1)

        x_union = torch.cat([x_out_2, x_out_1], dim=-1)
        x_union_expand = x_union.repeat(1, 1, 2)

        # mask_union = torch.ones_like(x_union_expand)
        # mask_union[:, entity_num: , :].zeros_()
        # x_union_expand = torch.mul(x_union_expand, mask_union)

        x_kr = torch.mul(x_union_expand, x_sum)
        x_kr = x_kr.view(bsz * imageNum, 4, dModel)
        # x_kr = x_kr.permute(0, 2, 1)
        x_kr = F.normalize(x_kr, dim=-1)

        x_out_q = self.conv1d_q(x_kr)
        x_out_q = x_out_q.squeeze(1)
        x_out_q = x_out_q.view(bsz, imageNum, dModel)

        x_out_s = self.conv1d_s(x_kr)
        x_out_s = x_out_s.squeeze(1)
        x_out_s = x_out_s.view(bsz, imageNum, dModel)
        
        return x_out_q, x_out_s


class SemanLCGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.build_loc_ctx_init()
        self.build_extract_textual_command()
        self.build_propagate_message()

    def build_loc_ctx_init(self):
        assert cfg.STEM_LINEAR != cfg.STEM_CNN
        if cfg.STEM_LINEAR:
            self.initKB = ops.Linear(cfg.D_FEAT, cfg.CTX_DIM)
            self.x_loc_drop = nn.Dropout(1 - cfg.stemDropout)
        elif cfg.STEM_CNN:
            self.cnn = nn.Sequential(
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.D_FEAT, cfg.STEM_CNN_DIM, (3, 3), padding=1),
                nn.ELU(),
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.STEM_CNN_DIM, cfg.CTX_DIM, (3, 3), padding=1),
                nn.ELU())

        self.initMem = nn.Parameter(torch.randn(1, 1, cfg.CTX_DIM))

    def build_extract_textual_command(self):
        self.qInput = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
        for t in range(cfg.MSG_ITER_NUM):
            qInput_layer2 = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
            setattr(self, "qInput%d" % t, qInput_layer2)
        self.cmd_inter2logits = ops.Linear(cfg.CMD_DIM, 1)

    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - cfg.readDropout)
        self.project_x_loc = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.project_x_ctx = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.queries = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.keys = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.vals = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.proj_keys = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.proj_vals = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.mem_update = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)
        self.combine_kb = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)

    def forward(self, images, seman_outputs, batch_size,
                entity_num):
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(images)
        for t in range(cfg.MSG_ITER_NUM):
            x_ctx = self.run_message_passing_iter(
                seman_outputs, x_loc, x_ctx,
                x_ctx_var_drop, entity_num, t)
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))
        return x_out

    def extract_textual_command(self, q_encoding, t):
        qInput_layer2 = getattr(self, "qInput%d" % t)
        act_fun = ops.activations[cfg.CMD_INPUT_ACT]
        cmd = qInput_layer2(act_fun(self.qInput(q_encoding))) # 128 * 512
        return cmd

    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint) # 128 * 49 * 512
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :] # 128 * 49 * 512
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :] # 128 * 49 * 512
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM)) # 128 * 49 * 49
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new

    def run_message_passing_iter(
            self, seman_outputs, x_loc, x_ctx,
            x_ctx_var_drop, entity_num, t):
        cmd = self.extract_textual_command(
                seman_outputs, t)
        x_ctx = self.propagate_message(
            cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num)
        return x_ctx

    def loc_ctx_init(self, images):
        if cfg.STEM_NORMALIZE:
            images = F.normalize(images, dim=-1)
        if cfg.STEM_LINEAR:
            x_loc = self.initKB(images) # 128 * 49 * 512
            x_loc = self.x_loc_drop(x_loc)
        elif cfg.STEM_CNN:
            images = torch.transpose(images, 1, 2)  # N(HW)C => NC(HW)
            x_loc = images.view(-1, cfg.D_FEAT, cfg.H_FEAT, cfg.W_FEAT)
            x_loc = self.cnn(x_loc)
            x_loc = x_loc.view(-1, cfg.CTX_DIM, cfg.H_FEAT * cfg.W_FEAT)
            x_loc = torch.transpose(x_loc, 1, 2)  # NC(HW) => N(HW)C
        if cfg.STEM_RENORMALIZE:
            x_loc = F.normalize(x_loc, dim=-1)

        x_ctx = self.initMem.expand(x_loc.size()) # 128 * 49 * 512
        x_ctx_var_drop = ops.generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(cfg.memoryDropout if self.training else 1.))
        return x_loc, x_ctx, x_ctx_var_drop
