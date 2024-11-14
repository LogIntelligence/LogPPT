from transformers import (
    RobertaForMaskedLM,
    AutoTokenizer,
)
import re
import torch
from torch import nn

from logppt.models.base import ModelBase
from logppt.postprocess import correct_single_template
import pdb
# from logppt.models.crf_layer import CRFLayer
from torchcrf import CRF


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
                 **kwargs
                 ):
        super().__init__(model_path, num_label_tokens)
        self.plm = RobertaForMaskedLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.use_crf = False
        self.tokenizer.model_max_length = self.plm.config.max_position_embeddings - 2
        self.vtoken = vtoken
        self.vtoken_id = self.tokenizer.convert_tokens_to_ids(vtoken)
        if kwargs.get("use_crf", True):
            self.use_crf = True
            self.crf = CRF(2, batch_first=True)
        self.crf_weight = kwargs.get("crf_weight", 0.1)

    def forward(self, batch):
        tags =  batch.pop('ori_labels', 'not found ner_labels')
        outputs = self.plm(**batch, output_hidden_states=True)
        plm_loss = outputs.loss
        if not self.use_crf:
            return plm_loss
        # logits = torch.softmax(outputs.logits, -1)
        logits = outputs.logits[:,:,[self.vtoken_id]]
        O_logits = outputs.logits[:,:,:self.vtoken_id].max(-1)[0].unsqueeze(-1)
        logits = torch.cat([O_logits, logits], dim=-1)
        logits = logits[:, 1:, :]
        # tags = batch['labels']
        mask = batch['attention_mask'].bool()
        mask[tags == -100] = 0
        mask = mask[:, 1:]
        tags = tags[:, 1:]
        # mask[:, 0] = 0
        # mask = mask.bool()
        tags = tags.masked_fill_(~mask, 0)
        # pdb.set_trace()
        crf_loss = self.crf(logits, tags, mask=mask)
        loss = plm_loss - self.crf_weight * crf_loss
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
        predictions = outputs.logits.argmax(dim=-1)
        vtoken_id = self.tokenizer.convert_tokens_to_ids(vtoken)

        if self.use_crf:
            logits = outputs.logits[:,:,[self.vtoken_id]]
            O_logits = outputs.logits[:,:,:self.vtoken_id].max(-1)[0].unsqueeze(-1)
            logits = torch.cat([O_logits, logits], dim=-1)
            logits = logits[:, 1:-1, :]
        
        # input_ids = tokenized_input['input_ids'][0]
        input_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])[1:-1]
        if self.use_crf:
            labels = self.crf.decode(logits)[0]
            print(input_tokens)
            print(labels)
            template = map_template(input_tokens, labels)
            print(template)
            # pdb.set_trace()
            return correct_single_template(template)
        else:
            logits = predictions.detach().cpu().clone().tolist()
            labels = [1 if x == vtoken_id else 0 for x in logits[0][1:-1]]
            template = map_template(input_tokens, labels)
            return correct_single_template(template)
    
    def load_checkpoint(self, checkpoint_path):
        self.plm = RobertaForMaskedLM.from_pretrained(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)


def get_label_with_crf(emissions=None, viterbi_decoder=None):
    emissions = emissions[0][1:-1, :]
    sent_scores = torch.tensor(emissions)
    sent_len, n_label = sent_scores.shape
    sent_probs = torch.nn.functional.softmax(sent_scores, dim=1)
    start_probs = torch.zeros(sent_len) + 1e-6
    sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
    feats = viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label + 1))
    print(feats.shape)
    pdb.set_trace()
    vit_labels = viterbi_decoder.viterbi(feats)
    vit_labels = vit_labels.view(sent_len)
    vit_labels = vit_labels.detach().cpu().numpy()
    return vit_labels

def map_template(inputs, labels):
    res = [" "]
    for i in range(0, len(inputs)):
        if labels[i] == 0:
            res.append(inputs[i])
        else:
            if "Ġ" in inputs[i]:
                # if "<*>" not in res[-1]:
                res.append("Ġ<*>")
            elif "<*>" not in res[-1]:
                res.append("<*>")
    r = "".join(res)
    r = r.replace("Ġ", " ")
    return r.strip()
