from transformers import (
    RobertaForMaskedLM,
    AutoTokenizer,
)
from collections import Counter
from logppt.models.base import ModelBase
import re
import torch
from torch import nn

from logppt.postprocess import correct_single_template

delimiters = "([ |\(|\)|\[|\]|\{|\})])"


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class RobertaForLogParsing(ModelBase):
    def __init__(self,
                 model_path,
                 num_label_tokens: int = 1,
                 vtoken="virtual-param",
                 ct_loss_weight=0.1,
                 **kwargs
                 ):
        super().__init__(model_path, num_label_tokens)
        self.plm = RobertaForMaskedLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.model_max_length = self.plm.config.max_position_embeddings - 2
        self.vtoken = vtoken
        self.sim = Similarity(0.05)
        self.ct_loss = nn.CrossEntropyLoss()
        self.ct_loss_weight = ct_loss_weight

    def compute_loss(self, batch, outputs):
        vtoken_id = self.tokenizer.convert_tokens_to_ids(self.vtoken)
        y = batch['labels'] == vtoken_id
        vtoken_reprs = outputs.hidden_states[-1][y]
        if vtoken_reprs.size(0) % 2 == 1:
            vtoken_reprs = vtoken_reprs[:-1]
        if vtoken_reprs.size(0) == 0:
            return outputs.loss
        z1_embed = vtoken_reprs[:vtoken_reprs.size(0) // 2]
        z2_embed = vtoken_reprs[vtoken_reprs.size(0) // 2:]
        sim = self.sim(z1_embed.unsqueeze(1), z2_embed.unsqueeze(0))
        labels = torch.arange(sim.size(0)).long().to(sim.device)
        loss_ct = self.ct_loss(sim, labels)
        return outputs.loss + loss_ct * self.ct_loss_weight

    def forward(self, batch, contrastive=False):
        outputs = self.plm(**batch, output_hidden_states=True)
        if contrastive:
            loss = self.compute_loss(batch, outputs)
        else:
            loss = outputs.loss
        return loss

    def add_label_token(self, label_map=None, num_tokens: int = 1):
        if label_map is None:
            crr_tokens, _ = self.plm.roberta.embeddings.word_embeddings.weight.shape
            self.plm.resize_token_embeddings(crr_tokens + num_tokens)

        sorted_add_tokens = sorted(
            list(label_map.keys()), key=lambda x: len(x), reverse=True)
        num_tokens, _ = self.plm.roberta.embeddings.word_embeddings.weight.shape
        self.plm.resize_token_embeddings(len(sorted_add_tokens) + num_tokens)
        for token in sorted_add_tokens:
            if token.startswith('i-'):
                index = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(token))
                if len(index) > 1:
                    raise RuntimeError(f"{token} wrong split: {index}")
                else:
                    index = index[0]
                # assert index >= num_tokens, (index, num_tokens, token)
                # n_label_word = len(list(set(label_map[token])))
                ws = label_map[token]
                e_token = self.plm.roberta.embeddings.word_embeddings.weight.data[ws[0]]
                for i in ws[1:]:
                    e_token += self.plm.roberta.embeddings.word_embeddings.weight.data[i]
                e_token /= len(ws)
                self.plm.roberta.embeddings.word_embeddings.weight.data[index] = e_token

    def named_parameters(self):
        return self.plm.named_parameters()

    def eval(self):
        self.plm.eval()

    def parse(self, log_line, device="cpu", vtoken="<*>"):
        def tokenize(log_line, max_length=256):
            log_tokens = re.split(delimiters, log_line)
            log_tokens = [token for token in log_tokens if len(token) > 0]
            refined_tokens = []
            if log_tokens[0] != " ":
                refined_tokens.append(log_tokens[0])
            for i in range(1, len(log_tokens)):
                if log_tokens[i] == " ":
                    continue
                if log_tokens[i - 1] == " ":
                    refined_tokens.append(" " + log_tokens[i])
                else:
                    refined_tokens.append(log_tokens[i])
            token_ids = []
            for token in refined_tokens:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                token_ids.extend(ids)
            token_ids = token_ids[:max_length - 2]
            token_ids = [self.tokenizer.bos_token_id] + \
                token_ids + [self.tokenizer.eos_token_id]
            return {
                'input_ids': torch.tensor([token_ids], dtype=torch.int64),
                'attention_mask': torch.tensor([[1] * len(token_ids)], dtype=torch.int64)
            }
        tokenized_input = tokenize(log_line)
        tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
        with torch.no_grad():
            outputs = self.plm(**tokenized_input, output_hidden_states=True)
        logits = outputs.logits.argmax(dim=-1)
        # logits = accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
        logits = logits.detach().cpu().clone().tolist()
        template = map_template(
            self.tokenizer, tokenized_input['input_ids'][0], logits[0], vtoken=vtoken)
        return correct_single_template(template)
    
    def load_checkpoint(self, checkpoint_path):
        self.plm = RobertaForMaskedLM.from_pretrained(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)


def map_template(tokenizer, c, t, vtoken="<*>"):
    vtoken = tokenizer.convert_tokens_to_ids(vtoken)
    tokens = tokenizer.convert_ids_to_tokens(c)
    res = [" "]
    for i in range(1, len(c)):
        if c[i] == tokenizer.sep_token_id:
            break
        if t[i] < vtoken:
            res.append(tokens[i])
        else:
            if "Ġ" in tokens[i]:
                # if "<*>" not in res[-1]:
                res.append("Ġ<*>")
            elif "<*>" not in res[-1]:
                res.append("<*>")
    r = "".join(res)
    r = r.replace("Ġ", " ")
    return r.strip()
