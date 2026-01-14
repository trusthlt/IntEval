import torch
import numpy as np
import json
import torch
from torch import nn

import os

class LimePredictor:
    def __init__(self, parameters, model, dataloader, seq_length, config_obj):
        self.parameters = parameters
        self.model = model
        self.dataloader = dataloader
        self.config = config_obj

    def create_chunks(self, perturbed_text):
        all_splits_words = list()
        all_splits_num = list()

        if len(perturbed_text.split()) == 0:
            perturbed_text = "[CLS] [SEP]"

        tokenized_text = perturbed_text.split()

        text = self.dataloader.get_text_from_tokens_wp_tokenizer(tokenized_text)
        words, num_splits, splits_counts = self.dataloader.split_sample_legal_bert(text)
        for sample in range(num_splits):
            splits_words = [''.join(w['tokens']) for w in words if sample in w['splits']]
            all_splits_words.append(splits_words)
        all_splits_num.append(num_splits)

        extra_info = [{'num_splits': num_splits}]

        tokenized_inputs = self.dataloader.tokenizer(
            all_splits_words, return_tensors='pt', max_length=self.parameters["context_size"],
            truncation=True, is_split_into_words=True, padding='max_length'
        )

        tokenized_inputs['extra_info'] = extra_info
        return tokenized_inputs

    def convert_text_to_features(self, text):

        if len(text.split()) == 0:
            text = "[CLS] [SEP]"

        input_ids = torch.tensor(self.dataloader.tokenizer.convert_tokens_to_ids(text.split())).unsqueeze(0)  # b x s
        # there was a query part, but we don't care, because it is not important for use :)
        token_type_ids = (input_ids != self.dataloader.tokenizer.pad_token_id).long()
        attention_mask = token_type_ids.clone()

        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}

    def predictor(self, text):

        examples = []

        for example in text:
            examples.append(self.convert_text_to_features(example))

        results = []

        input_data = list()
        for example in text:
            input_for_instance = self.create_chunks(example)
            input_data.append(input_for_instance)


        for instance in input_data:

            batch = {
                "input_ids": instance["input_ids"].to(self.config.device),
                "token_type_ids": instance["token_type_ids"].to(self.config.device),
                "attention_mask": instance["attention_mask"].to(self.config.device),
                "retain_gradient": False,
                "inputs_embeds": None,
                "extra_info": instance["extra_info"]
            }

            with torch.no_grad():
                logits, _ = self.model(**batch)

            pred_dist = torch.softmax(logits, dim=1)
            results.append(pred_dist.cpu().detach().numpy()[0])

        results_array = np.array(results)

        return results_array

class ShapleyModelWrapper(nn.Module):

    def __init__(self, model):
        super(ShapleyModelWrapper, self).__init__()

        self.model = model

    def forward(self, embeddings):
        head_mask = [None] * self.model.core_model.model.config.num_hidden_layers

        encoder_outputs = self.model.core_model.model.encoder(
            embeddings,
            head_mask=head_mask,
            output_attentions=self.model.core_model.model.config.output_attentions,
            output_hidden_states=self.model.core_model.model.config.output_attentions,
            return_dict=self.model.core_model.model.config.return_dict
        )

        num_splits = int(embeddings.shape[0] / 2)
        split_sizes = [num_splits] * 2
        doc_representation = torch.zeros(2, self.model.core_model.model.config.hidden_size,
                                         device=self.model.config_obj.device)

        sequence_output = encoder_outputs[0][:, 0, :]

        init_idx = 0
        for idx, splits in enumerate(split_sizes):
            doc_embedding = sequence_output[init_idx: (init_idx + splits), :]

            aggregated_embeddings = self.model.aggregator(doc_embedding)
            doc_embedding = aggregated_embeddings.unsqueeze(0)

            doc_representation[idx] = doc_embedding.squeeze(0)
            init_idx += num_splits
            torch.cuda.empty_cache()

        logits = self.model.classifier(doc_representation)

        return logits