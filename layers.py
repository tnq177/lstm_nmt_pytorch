import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class UniLSTMEncoder(nn.Module):
    """Unidirectional LSTM Encoder"""
    def __init__(self, args):
        super(UniLSTMEncoder, self).__init__()
        self.args = args
        self.num_layers = args.num_enc_layers
        self.embed_dim = args.embed_dim
        self.dropout = args.dropout
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.embed_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True)

        for p in self.lstm.parameters():
            nn.init.uniform_(p, a=-0.01, b=0.01)

    def forward(self, src_inputs):
        # x: [bsz, seq_len, embed_dim]
        src_inputs = F.dropout(src_inputs, p=self.dropout, training=self.training)
        x, _ = self.lstm(src_inputs) # [h0, c0] both zeros
        # last LSTM output is not dropped out so we need to do it
        return F.dropout(x, p=self.dropout, training=self.training) + src_inputs


class BilinearAttention(nn.Module):
    """BilinearAttention https://arxiv.org/abs/1508.04025"""
    def __init__(self, args):
        super(BilinearAttention, self).__init__()
        self.args = args
        self.embed_dim = args.embed_dim
        self.dropout = args.att_dropout
        self.weight = Parameter(torch.Tensor(self.embed_dim, self.embed_dim))

        nn.init.uniform_(self.weight, a=-0.01, b=0.01)

    def forward(self, q, k, v, mask):
        # q: [bsz, tgt_len, embed_dim]
        # k: [bsz, src_len, embed_dim]
        # v: [bsz, src_len, embed_dim]
        # [bsz, tgt_len, embed_dim] x [bsz, embed_dim, src_len] -> [bsz, tgt_len, src_len]
        att_weights = torch.bmm(q, k)
        att_weights.masked_fill_(mask, -1e9)
        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
        # [bsz, tgt_len, src_len] x [bsz, src_len, embed_dim] = [bsz, tgt_len, embed_dim]
        context = torch.bmm(att_weights, v)
        return context

    def proj_k(self, k):
        return F.linear(k, self.weight).transpose(1, 2)


class LSTMDecoder(nn.Module):
    """LSTMDecoder with feed-input https://arxiv.org/abs/1508.04025"""
    def __init__(self, args):
        super(LSTMDecoder, self).__init__()
        self.args = args
        self.num_layers = args.num_dec_layers
        self.embed_dim = args.embed_dim
        self.dropout = args.dropout
        self.lstm = nn.LSTM(
            input_size=self.embed_dim * 2,
            hidden_size=self.embed_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True)
        self.attention = BilinearAttention(args)
        self.output_weight = Parameter(torch.Tensor(self.embed_dim, 2 * self.embed_dim))
        if self.args.fix_norm:
            self.scale = ScaleNorm(self.embed_dim ** 0.5)
        else:
            self.scale = None

        nn.init.uniform_(self.output_weight, a=-0.01, b=0.01)
        for p in self.lstm.parameters():
            nn.init.uniform_(p, a=-0.01, b=0.01)

    def forward(self, tgt_inputs, encoder_out, encoder_mask):
        # encoder_out: [bsz, src_len, dim]
        # encoder_mask: [bsz, 1, src_len]
        # tgt_inputs: [bsz, tgt_len, dim]
        bsz, tgt_len, embed_dim = tgt_inputs.size()
        tgt_inputs = F.dropout(tgt_inputs, p=self.dropout, training=self.training)
        h = torch.zeros((self.num_layers, bsz, self.embed_dim)).type(tgt_inputs.type())
        c = torch.zeros((self.num_layers, bsz, self.embed_dim)).type(tgt_inputs.type())
        prev_h_tilde = torch.zeros((bsz, 1, self.embed_dim)).type(tgt_inputs.type())
        all_outputs = []
        # [bsz, dim, src_len]
        encoder_k = self.attention.proj_k(encoder_out)
        # [bsz, src_len, dim]
        encoder_v = encoder_out
        for i in range(tgt_len):
            # [bsz, 1, dim]
            x = tgt_inputs[:, i:i+1, :]
            # feed input --> [bsz, 1, dim + dim]
            x = torch.cat((x, prev_h_tilde), dim=-1)
            x, (h, c) = self.lstm(x, (h, c))
            x = F.dropout(x, p=self.dropout, training=self.training)
            context = self.attention(x, encoder_k, encoder_v, encoder_mask)
            # [bsz, 1, dim + dim]
            x = torch.cat((x, context), dim=-1)
            x = torch.tanh(F.linear(x, self.output_weight))
            prev_h_tilde = x
            all_outputs.append(x)

        # [bsz, tgt_len, dim]
        all_outputs = torch.cat(all_outputs, dim=1)
        all_outputs = F.dropout(all_outputs, p=self.dropout, training=self.training)
        if self.scale is not None:
            all_outputs = self.scale(all_outputs)
        return all_outputs

    def beam_step(self, inp, cache):
        bsz, beam_size, embed_dim = inp.size()
        # [bsz, beam, 1, src_len] -> [bsz x beam, 1, src_len]
        encoder_mask = cache['encoder_mask'].reshape(bsz * beam_size, 1, -1)
        # [bsz, beam, dim, src_len] -> [bsz x beam, dim, src_len]
        encoder_k = cache['encoder_k'].reshape(bsz * beam_size, embed_dim, -1)
        # [bsz, beam, src_len, dim] --> [bsz x beam, src_len, dim]
        encoder_v = cache['encoder_v'].reshape(bsz * beam_size, -1, embed_dim)
        # [bsz, beam, num_layers, dim] -> [bsz x beam, num_layers, dim] --> [num_layers, bsz x beam, dim]
        h = cache['h'].reshape(bsz * beam_size, -1, embed_dim).transpose(0, 1).contiguous()
        c = cache['c'].reshape(bsz * beam_size, -1, embed_dim).transpose(0, 1).contiguous()
        # [bsz, beam, 1, dim] --> [bsz x beam, 1, dim]
        prev_h_tilde = cache['prev_h_tilde'].reshape(bsz * beam_size, 1, embed_dim)

        x = inp.reshape(bsz * beam_size, 1, embed_dim)
        # [bsz x beam, 1, dim * 2]
        x = torch.cat((x, prev_h_tilde), dim=-1)
        x, (h, c) = self.lstm(x, (h, c))
        context = self.attention(x, encoder_k, encoder_v, encoder_mask)
        x = torch.cat((x, context), dim=-1)
        x = torch.tanh(F.linear(x, self.output_weight))
        prev_h_tilde = x
        if self.scale is not None:
            x = self.scale(x)

        cache['h'] = h.transpose(0, 1).reshape(bsz, beam_size, -1, embed_dim)
        cache['c'] = c.transpose(0, 1).reshape(bsz, beam_size, -1, embed_dim)
        cache['prev_h_tilde'] = prev_h_tilde.reshape(bsz, beam_size, 1, embed_dim)
        return x.reshape(bsz * beam_size, embed_dim)

    def beam_decode(self, encoder_out, encoder_mask, get_input_fn, logprob_fn, bos_id, eos_id, max_len, beam_size=4, alpha=-1):
        bsz, _, embed_dim = encoder_out.size()
        # initial beam=1
        # [bsz x beam, 1, src_len] --> [bsz, beam, 1, src_len]
        cache = {}
        cache['encoder_mask'] = encoder_mask.reshape(bsz, 1, 1, -1)
        # [bsz x beam, dim, src_len] --> [bsz, beam, dim, src_len]
        cache['encoder_k'] = self.attention.proj_k(encoder_out).reshape(bsz, 1, embed_dim, -1)
        # [bsz x beam, src_len, dim] --> [bsz, beam, src_len, dim]
        cache['encoder_v'] = encoder_out.reshape(bsz, 1, -1, embed_dim)
        cache['h'] = torch.zeros((bsz, 1, self.num_layers, embed_dim)).type(encoder_out.type())
        cache['c'] = torch.zeros((bsz, 1, self.num_layers, embed_dim)).type(encoder_out.type())
        cache['prev_h_tilde'] = torch.zeros((bsz, 1, 1, embed_dim)).type(encoder_out.type())

        # First step
        inp = torch.tensor([bos_id] * bsz).reshape(bsz, 1)
        # [bsz, beam, dim]
        inp = get_input_fn(inp)
        # [bsz * beam, dim]
        y = self.beam_step(inp, cache)
        # [bsz * beam, V]
        probs = logprob_fn(y)
        # no eos initially
        probs[:, eos_id] = float('-inf')
        # [bsz, beam], [bsz, beam]
        all_probs, symbols = torch.topk(probs, beam_size, dim=-1)

        # Expand by beam
        # beam != 1 from now on
        cache['encoder_mask'] = cache['encoder_mask'].expand(-1, beam_size, -1, -1)
        cache['encoder_k'] = cache['encoder_k'].expand(-1, beam_size, -1, -1)
        cache['encoder_v'] = cache['encoder_v'].expand(-1, beam_size, -1, -1)
        cache['h'] = cache['h'].expand(-1, beam_size, -1, -1)
        cache['c'] = cache['c'].expand(-1, beam_size, -1, -1)
        cache['prev_h_tilde'] = cache['prev_h_tilde'].expand(-1, beam_size, -1, -1)

        last_probs = all_probs.reshape(bsz, beam_size)
        last_scores = last_probs.clone()
        all_symbols = symbols.reshape(bsz, beam_size, 1)

        num_classes = probs.size(-1)
        # convert to eos_mask.type() to be on the same device
        not_eos_mask = (torch.arange(num_classes).reshape(1, -1) != eos_id).type(encoder_mask.type())
        maximum_length = max_len.max().item()
        ret = [None] * bsz
        batch_idxs = torch.arange(bsz)
        for time_step in range(1, maximum_length + 1):
            surpass_length = (time_step > max_len) + (time_step == maximum_length)
            finished_decoded = torch.sum((all_symbols[:, :, -1] == eos_id).type(max_len.type()), dim=-1) == beam_size
            finished_sents = surpass_length + finished_decoded
            if finished_sents.any():
                # Trimp finished within batch
                for j in range(finished_sents.size(0)):
                    if finished_sents[j]:
                        ret[batch_idxs[j]] = {
                            'symbols': all_symbols[j].clone(),
                            'probs': last_probs[j].clone(),
                            'scores': last_scores[j].clone()
                        }

                all_symbols = all_symbols[~finished_sents]
                last_probs = last_probs[~finished_sents]
                last_scores = last_scores[~finished_sents]
                max_len = max_len[~finished_sents]
                batch_idxs = batch_idxs[~finished_sents]

                cache['encoder_mask'] = cache['encoder_mask'][~finished_sents]
                cache['encoder_k'] = cache['encoder_k'][~finished_sents]
                cache['encoder_v'] = cache['encoder_v'][~finished_sents]
                cache['h'] = cache['h'][~finished_sents]
                cache['c'] = cache['c'][~finished_sents]
                cache['prev_h_tilde'] = cache['prev_h_tilde'][~finished_sents]


            if finished_sents.all():
                break

            # bsz could've changed after trimming
            bsz = all_symbols.size(0)
            # [bsz, beam] --> [bsz, beam, dim]
            last_symbols = all_symbols[:, :, -1]
            inps = get_input_fn(last_symbols)
            # [bsz x beam, dim]
            ys = self.beam_step(inps, cache)
            probs = logprob_fn(ys) # [bsz x beam, V]
            # [bsz x beam, 1]
            last_probs = last_probs.reshape(-1, 1)
            last_scores = last_scores.reshape(-1, 1)
            length_penalty = 1.0 if alpha == -1 else (5.0 + time_step + 1.0) ** alpha / 6.0 ** alpha
            finished_mask = last_symbols.reshape(-1) == eos_id
            beam_probs = probs.clone()
            if finished_mask.any():
                beam_probs[finished_mask] = last_probs[finished_mask].expand(-1, num_classes).masked_fill(not_eos_mask, float('-inf'))
                beam_probs[~finished_mask] = last_probs[~finished_mask] + probs[~finished_mask]
            else:
                beam_probs = last_probs + probs

            beam_scores = beam_probs.clone()
            if finished_mask.any():
                beam_scores[finished_mask] = last_scores[finished_mask].expand(-1, num_classes).masked_fill(not_eos_mask, float('-inf'))
                beam_scores[~finished_mask] = beam_probs[~finished_mask] / length_penalty
            else:
                beam_scores = beam_probs / length_penalty

            beam_probs = beam_probs.reshape(bsz, -1)
            beam_scores = beam_scores.reshape(bsz, -1)
            # [bsz, beam], [bsz, beam] (top beam from beam * num_classes along dim=-1)
            max_scores, idxs = torch.topk(beam_scores, beam_size, dim=-1)
            parent_idxs = idxs // num_classes
            symbols = (idxs - parent_idxs * num_classes).type(idxs.type())

            # gather along dim=-1 at idxs
            last_probs = torch.gather(beam_probs, -1, idxs)
            last_scores = max_scores
            offset = torch.arange(bsz).reshape(-1, 1).type(parent_idxs.type()) * beam_size
            # [bsz, beam] + [bsz, 1] --> [bsz x beam]
            parent_idxs = (parent_idxs + offset).reshape(-1)
            all_symbols = all_symbols.reshape(bsz * beam_size, -1)[parent_idxs].reshape(bsz, beam_size, -1)
            # [bsz, beam, len] (+) [bsz, beam, 1] --> [bsz, beam, len+1]
            all_symbols = torch.cat((all_symbols, symbols.unsqueeze(-1)), -1)

            cache['h'] = cache['h'].reshape(bsz * beam_size, -1, embed_dim)[parent_idxs].reshape(bsz, beam_size, -1, embed_dim)
            cache['c'] = cache['c'].reshape(bsz * beam_size, -1, embed_dim)[parent_idxs].reshape(bsz, beam_size, -1, embed_dim)
            cache['prev_h_tilde'] = cache['prev_h_tilde'].reshape(bsz * beam_size, -1, embed_dim)[parent_idxs].reshape(bsz, beam_size, -1, embed_dim)

        if batch_idxs.size(0) > 0:
            for j in range(batch_size.size(0)):
                ret[batch_idxs[j]] = {
                    'symbols': all_symbols[j].clone(),
                    'probs': last_probs[j].clone(),
                    'scores': last_scores[j].clone()
                }

        return ret
