import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .config import cfg

def apply_mask1d(attention, image_locs):
    batch_size, num_loc = attention.size()
    tmp1 = attention.new_zeros(num_loc)
    tmp1[:num_loc] = torch.arange(
        0, num_loc, dtype=attention.dtype).unsqueeze(0)

    tmp1 = tmp1.expand(batch_size, num_loc)
    tmp2 = image_locs.type(tmp1.type())
    tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
    mask = torch.ge(tmp1, tmp2)
    attention = attention.masked_fill(mask, -1e30)
    return attention
    
class Transformer(nn.Module):
    def __init__(self, ninp=300, nhead=4, nhid=1024, nlayers=2, dropout=0.3, mask_len=12):
        super(Transformer, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.mask_len = mask_len
        self.nhead = nhead

    def _generate_square_subsequent_mask(self, x_length):
        bsz = x_length.size(0)

        mask = (torch.ones(bsz * self.nhead, self.mask_len, self.mask_len) == 1)
        nhead_index = 0 
        for i in range(bsz * self.nhead):
            mask[i, x_length[nhead_index]: self.mask_len, :] = False
            mask[i, :, x_length[nhead_index]: self.mask_len] = False
            if (i + 1) % self.nhead == 0:
                nhead_index += 1

        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask

    def forward(self, x_emb, x_length):
        #mask = self._generate_square_subsequent_mask(x_length)
        #mask = mask.cuda()
        x_emb = x_emb.permute(1, 0, 2)
        output = self.transformer_encoder(x_emb)
        return output


class GatedTrans(nn.Module):
    """docstring for GatedTrans"""
    def __init__(self, in_dim, out_dim):
        super(GatedTrans, self).__init__()
        
        self.embed_y = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim
            ),
            nn.Tanh()
        )
        self.embed_g = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim
            ),
            nn.Sigmoid()
        )

    def forward(self, x_in):
        x_y = self.embed_y(x_in)
        x_g = self.embed_g(x_in)
        x_out = x_y*x_g

        return x_out


class Q_ATT(nn.Module):
    """Self attention module of questions."""
    def __init__(self, config):
        super(Q_ATT, self).__init__()

        self.embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["lstm_hidden_size"]*2,
                config["lstm_hidden_size"]
            ),
        )        
        self.att = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                config["lstm_hidden_size"],
                1
            )
        )
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, ques_word, ques_word_encoded, ques_not_pad):
        # ques_word shape: (batch_size, num_rounds, quen_len_max, word_embed_dim)
        # ques_embed shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size * 2)
        # ques_not_pad shape: (batch_size, num_rounds, quen_len_max)
        # output: img_att (batch_size, num_rounds, embed_dim)
        batch_size = ques_word.size(0)
        num_rounds = ques_word.size(1)
        quen_len_max = ques_word.size(2)

        ques_embed = self.embed(ques_word_encoded) # shape: (batch_size, num_rounds, quen_len_max, embed_dim)
        ques_norm = F.normalize(ques_embed, p=2, dim=-1) # shape: (batch_size, num_rounds, quen_len_max, embed_dim) 
        
        att = self.att(ques_norm).squeeze(-1) # shape: (batch_size, num_rounds, quen_len_max)
        # ignore <pad> word
        att = self.softmax(att)
        att = att*ques_not_pad # shape: (batch_size, num_rounds, quen_len_max)
        att = att / torch.sum(att, dim=-1, keepdim=True) # shape: (batch_size, num_rounds, quen_len_max)
        feat = torch.sum(att.unsqueeze(-1) * ques_word, dim=-2) # shape: (batch_size, num_rounds, rnn_dim)
        
        return feat, att


class Encoder(nn.Module):
    def __init__(self, embInit):
        super().__init__()
        self.embeddingsVar = nn.Parameter(
            torch.Tensor(embInit), requires_grad=(not cfg.WRD_EMB_FIXED))
        self.enc_input_drop = nn.Dropout(1 - cfg.encInputDropout)
        self.rnn0 = BiLSTM()
        self.question_drop = nn.Dropout(1 - cfg.qDropout)

        self.enc_seman_drop = nn.Dropout(1 - cfg.qDropout)

        self.transformer = Transformer()

    def forward(self, qIndices, questionLengths, semanIndices, semanLengths):
        # Word embedding
        embeddingsVar = self.embeddingsVar.cuda()
        embeddings = torch.cat(
            [torch.zeros(1, cfg.WRD_EMB_DIM, device='cuda'), embeddingsVar],
            dim=0)
        questions = F.embedding(qIndices, embeddings) # 128 * 30 * 300
        questions = self.enc_input_drop(questions)

        semans = F.embedding(semanIndices, embeddings) # 128 * 30 * 300
        semans = self.enc_seman_drop(semans)

        # RNN (LSTM)
        questionCntxWords, vecQuestions = self.rnn0(questions, questionLengths) #128 * 30 * 512 128 * 512
        vecQuestions = self.question_drop(vecQuestions)

        # Transformer
        semanCnt = self.transformer(semans, semanLengths)

        return questionCntxWords, vecQuestions, semanCnt


class BiLSTM(nn.Module):
    def __init__(self, forget_gate_bias=1.):
        super().__init__()
        self.bilstm = torch.nn.LSTM(
            input_size=cfg.WRD_EMB_DIM, hidden_size=cfg.ENC_DIM // 2,
            num_layers=1, batch_first=True, bidirectional=True)

        d = cfg.ENC_DIM // 2

        # initialize LSTM weights (to be consistent with TensorFlow)
        fan_avg = (d*4 + (d+cfg.WRD_EMB_DIM)) / 2.
        bound = np.sqrt(3. / fan_avg) #0.0616236
        nn.init.uniform_(self.bilstm.weight_ih_l0, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_hh_l0, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_ih_l0_reverse, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_hh_l0_reverse, -bound, bound)

        # initialize LSTM forget gate bias (to be consistent with TensorFlow)
        self.bilstm.bias_ih_l0.data[...] = 0.
        self.bilstm.bias_ih_l0.data[d:2*d] = forget_gate_bias
        self.bilstm.bias_hh_l0.data[...] = 0.
        self.bilstm.bias_hh_l0.requires_grad = False
        self.bilstm.bias_ih_l0_reverse.data[...] = 0.
        self.bilstm.bias_ih_l0_reverse.data[d:2*d] = forget_gate_bias
        self.bilstm.bias_hh_l0_reverse.data[...] = 0.
        self.bilstm.bias_hh_l0_reverse.requires_grad = False

    def forward(self, questions, questionLengths):
        # sort samples according to question length (descending)
        sorted_lengths, indices = torch.sort(questionLengths, descending=True)
        sorted_questions = questions[indices]
        _, desorted_indices = torch.sort(indices, descending=False)# 128

        # pack questions for LSTM forwarding
        packed_questions = nn.utils.rnn.pack_padded_sequence(
            sorted_questions, sorted_lengths, batch_first=True)
        packed_output, (sorted_h_n, _) = self.bilstm(packed_questions) #s_h_n: 2 * 128 * 256
        sorted_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=questions.size(1)) # 128 * 30 * 512
        sorted_h_n = torch.transpose(sorted_h_n, 1, 0).reshape( 
            questions.size(0), -1) # 128 * 512

        # sort back to the original sample order
        output = sorted_output[desorted_indices] # 128 * 30 * 512
        h_n = sorted_h_n[desorted_indices] # 128 * 512

        return output, h_n
