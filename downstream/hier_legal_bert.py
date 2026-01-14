import torch
import torch.nn as nn
from typing import Any
from transformers import AutoModel, AutoConfig
import numpy as np

class HierarchicalLEGALBERT(nn.Module):
    def __init__(self, parameters, config_obj, num_labels):
        super(HierarchicalLEGALBERT, self).__init__()
        model_path = parameters['model_path']
        self.parameters_obj = parameters
        self.config_obj = config_obj
        self.bert_config = AutoConfig.from_pretrained(config_obj.model_names[model_path], output_attentions=True)
        self.core_model = BertWrapper(
            model=AutoModel.from_pretrained(config_obj.model_names[model_path], config=self.bert_config),
            device=self.config_obj.device
        )
        self.output_dim = num_labels
        self.dropout = nn.Dropout(p=0.1) # notice it must be 0.1 # also test without it (meaning setting it to 0)
        self.aggregator = TransformerAggregator(embed_dim=self.core_model.model.config.hidden_size, num_heads=8, num_layers=3, num_labels=num_labels)
        # self.label_attention = LabelAttention(num_labels, self.core_model.model.config.hidden_size)

        self.classifier = nn.Linear(self.core_model.model.config.hidden_size, num_labels)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.0)
        self.output = None
        self.weights_or = None
        self.weights = None
        self.approximation_error = None

    def process_extra_chunks(self, num_chunks, input_ids, attention_mask, token_type_ids, inputs_embeds, ig, retain_grad=False, max_size=10):
        all_size = num_chunks

        init = 0
        end = 0
        hidden_states = list()
        pooled_outputs = list()
        attention_list = list()
        for i in range(0, all_size, max_size):
            end = init + min(max_size, all_size - end)
            last_hidden_state, pooled_output, attentions = self.core_model(
                input_ids[init:end] if input_ids is not None else input_ids,
                attention_mask=attention_mask[init:end] if attention_mask is not None else attention_mask,
                token_type_ids=token_type_ids[init:end] if token_type_ids is not None else token_type_ids,
                inputs_embeds=inputs_embeds[init:end] if inputs_embeds is not None else inputs_embeds,
                ig=ig
            )
            hidden_states.append(last_hidden_state)
            pooled_outputs.append(pooled_output)
            attention_list.append(attentions)
            init = end
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        attentions_all = self.combine_attention_chunks(attention_list, retain_grad=retain_grad)
        hidden_states_data = torch.cat([self.dropout(h_state) for h_state in hidden_states])
        return hidden_states_data, pooled_outputs, attentions_all

    def combine_attention_chunks(self, list_of_attentions, extra_step=False, retain_grad=False):
        attentions_all = list()
        for meta_chunk in list_of_attentions:
            if not attentions_all:
                for chunk in meta_chunk:
                    if retain_grad:
                        chunk.retain_grad()
                    attentions_all.append(chunk)
                continue

            for idx, chunk in enumerate(meta_chunk):
                if retain_grad:
                    chunk.retain_grad()
                attentions_all[idx] = torch.cat([attentions_all[idx], chunk])

        return attentions_all

    def extra_chunk_processing(self, inputs, max_chunks, splits):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        inputs_embeds = inputs['inputs_embeds']
        num_of_chunks_in_batch = sum(splits)

        if num_of_chunks_in_batch > max_chunks:
            hidden_states = list()
            pooled_outputs = list()
            attention_list = list()
            init = 0
            for each in splits:
                end = init + each

                if each <= 10:
                    last_hidden_state, pooled_output, attentions = self.core_model(
                        input_ids[init:end] if input_ids is not None else input_ids,
                        attention_mask=attention_mask[init:end] if attention_mask is not None else attention_mask,
                        token_type_ids=token_type_ids[init:end] if token_type_ids is not None else token_type_ids,
                        inputs_embeds=inputs_embeds[init:end] if inputs_embeds is not None else inputs_embeds,
                        ig=inputs['ig']
                    )
                else:
                    last_hidden_state, pooled_output, attentions = self.process_extra_chunks(
                        each,
                        input_ids[init:end] if input_ids is not None else input_ids,
                        attention_mask=attention_mask[init:end] if attention_mask is not None else attention_mask,
                        token_type_ids=token_type_ids[init:end] if token_type_ids is not None else token_type_ids,
                        inputs_embeds=inputs_embeds[init:end] if inputs_embeds is not None else inputs_embeds,
                        ig=inputs['ig'],
                        retain_grad=inputs['retain_gradient']
                    )

                init = end


                hidden_states.append(last_hidden_state)
                pooled_outputs.append(pooled_output)
                attention_list.append(attentions)
                torch.cuda.empty_cache()

            hidden_states = torch.cat([self.dropout(h_state) for h_state in hidden_states])
            attention_weights = self.combine_attention_chunks(attention_list, extra_step=True, retain_grad=inputs['retain_gradient'])
            cls_embeddings = hidden_states[:, 0, :]
        else:
            last_hidden_state, pooled_outputs, attention_weights = self.core_model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                ig=inputs['ig']
            )

            hidden_states = self.dropout(last_hidden_state)
            cls_embeddings = hidden_states[:, 0, :]

        return cls_embeddings, hidden_states, pooled_outputs, attention_weights


    def forward(self, **inputs):
        if 'ig' not in inputs: inputs['ig'] = int(1)
        max_chunks = 8 if self.parameters_obj['model_path'] == 'legal_longformer' else 10000 # 30 in learning
        splits = [s['num_splits'] for s in inputs["extra_info"]]

        cls_embeddings, self.output, pooled_output, attention_weights = self.extra_chunk_processing(inputs, max_chunks, splits)


        # in the original implementation each input is in the shape of (batch size  * attention dim). However, in our
        # case shapes are: (number of chunks in the batch * attention dim)
        # only on inference part
        # if isinstance(attention_weights, torch.Tensor):
        self.weights_or = attention_weights[-1]
        if inputs['retain_gradient']:

            self.weights_or.retain_grad()
            self.core_model.word_embedds.retain_grad()
        self.weights = self.weights_or[:, :, 0, :].mean(1)


        doc_representation = torch.zeros(len(inputs["extra_info"]), self.core_model.model.config.hidden_size, device=self.config_obj.device) # add device later
        init_idx = 0
        for idx, sample_info in enumerate(inputs['extra_info']):
            num_splits = sample_info['num_splits']
            doc_embedding = cls_embeddings[init_idx: (init_idx + num_splits), :]

            aggregated_embeddings = self.aggregator(doc_embedding)
            doc_embedding = aggregated_embeddings.unsqueeze(0)

            doc_representation[idx] = doc_embedding.squeeze(0)
            init_idx += num_splits
            torch.cuda.empty_cache()

        logits = self.classifier(doc_representation)
        torch.cuda.empty_cache()

        return logits, self.weights

    def integrated_grads(self, original_grad, original_pred, steps=10, **inputs):
        """
        Method performs the integrated gradients of the model (inspired by ISR technique)
        :param original_grad:
        :param original_pred:
        :param steps:
        :param inputs:
        :return:
        """
        grad_list = [original_grad]

        pred = None
        baseline = None
        for x in torch.arange(start=0.0, end=1.0, step=(1.0 - 0.0) / steps):

            self.eval()
            self.zero_grad()

            inputs["ig"] = x

            pred, _ = self.forward(**inputs)

            if len(pred.shape) == 1:
                pred = pred.unsqueeze(0)

            rows = torch.arange(pred.size(0))

            if x == 0.0:
                baseline = pred[rows, original_pred[1]]

            pred[rows, original_pred[1]].sum().backward()

            embed_grad = self.core_model.model.embeddings.word_embeddings.weight.grad
            g = embed_grad[inputs["input_ids"].long()]

            grad_list.append(g)

        attributions = torch.stack(grad_list).mean(0)

        em = self.core_model.model.embeddings.word_embeddings.weight[inputs["input_ids"].long()]

        ig = (attributions * em).sum(-1)

        self.approximation_error = torch.abs((attributions.sum() - (original_pred[0] - baseline).sum()) / pred.size(0))

        return ig


class TransformerAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_labels):
        super(TransformerAggregator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, num_labels)

    def forward(self, x, mask=None):
        aggregated_output = self.transformer_encoder(x, src_key_padding_mask=mask)
        pooled_output = torch.mean(aggregated_output, dim=0)
        return pooled_output

# As employed by Chrysostomou et al., (2021) and adapted to LEGAL-BERT
class BertWrapper(nn.Module):
    def __init__(self, model: HierarchicalLEGALBERT, device: torch.device):
        super(BertWrapper, self).__init__()
        self.word_embedds = None
        self.model = model
        self.device = device

    def forward(self, input_sequence: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, ig: int = int(1), inputs_embeds: Any = None):
        """
        Method performs the forward pass of the wrapper
        :param input_sequence: input ids
        :param attention_mask: attention mask for tokens
        :param token_type_ids: token types
        :param ig: integrated gradient ratio
        :param inputs_embeds: input embeddings -> if it is None, we create it here
        :return: tuple containing the output of the model, pooled output and attention weights
        """
        if inputs_embeds is None:
            embeddings, self.word_embedds = self.bert_embeddings_creator(
                input_sequence,
                position_ids = None,
                token_type_ids = token_type_ids
            )
        else:
            embeddings = inputs_embeds
        assert 0. <= ig <= int(1), "IG ratio cannot be out of the range 0-1"

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.model.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.model.config.num_hidden_layers

        encoder_outputs = self.model.encoder(
            embeddings * ig,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_attentions,
            return_dict=self.model.config.return_dict
        )

        sequence_output = encoder_outputs[0]
        attentions = encoder_outputs[2]
        pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None

        return sequence_output, pooled_output, attentions

    def bert_embeddings_creator(self, input_sequence: torch.Tensor, position_ids: torch.Tensor=None, token_type_ids: torch.Tensor=None):
        """
        Method creates a BERT embeddings
        :param input_sequence:
        :param position_ids:
        :param token_type_ids:
        :return: tuple containing bert embeddings and word embeddings
        """
        if input_sequence is not None:
            input_shape = input_sequence.size()

        sequence_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(512).expand((1, -1)).to(self.device)
            position_ids = position_ids[:, :sequence_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long).to(self.device)

        embed = self.model.embeddings.word_embeddings(input_sequence)

        position_embeddings = self.model.embeddings.position_embeddings(position_ids)

        token_type_embeddings = self.model.embeddings.token_type_embeddings(token_type_ids)


        embeddings = embed + position_embeddings + token_type_embeddings
        embeddings = self.model.embeddings.LayerNorm(embeddings)
        embeddings = self.model.embeddings.dropout(embeddings)

        return embeddings, embed