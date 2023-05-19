# coding=utf-8
import math

import numpy as np
import torch

from transformers import T5ForConditionalGeneration
import torch.nn as nn
import torch.nn.functional
import warnings

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)


# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class Quad_t5_model(nn.Module):
    def __init__(self, config, model_name_or_path):
        super(Quad_t5_model, self).__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(config.p)
        self.mc_dropout_num = config.mc_forward_num
        self.config = config

    def use_mc_dropout(self, sequence_output):
        return self.dropout(sequence_output)

    def uncertainty_aware_predictions(self, sequence_output):
        all_logits = None

        for num in range(self.mc_dropout_num):
            # Sampling the mask matrix using dropout's built-in method
            temp = self.use_mc_dropout(sequence_output)

            if self.t5_model.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                temp = temp * (self.t5_model.model_dim ** -0.5)

            lm_logits = self.t5_model.lm_head(temp).unsqueeze(0)

            if all_logits is not None:
                all_logits = torch.cat((all_logits, lm_logits), dim=0)
            else:
                all_logits = lm_logits
        return all_logits

    def negative_samples_acquisition(self, all_logits, labels):
        all_c = self.MAX(all_logits)
        negative_samples_masks = self.get_negative_samples_masks(all_c, labels, all_logits)
        return negative_samples_masks

    @torch.no_grad()
    def MAX(self, all_logits, top_k=1):
        all_top_k_ids = torch.topk(all_logits, top_k, dim=-1)
        return all_top_k_ids.indices.squeeze(-1)

    @torch.no_grad()
    def get_negative_samples_masks(self, all_c, labels, all_logits):
        batch_size = labels.shape[0]
        all_lengths = (labels != 0).sum(-1)
        mask_results = torch.zeros_like(all_logits)
        for i in range(all_c.shape[0]):
            for j in range(batch_size):
                mask_results[i, j, :all_lengths[j]] = mask_results[i, j, :all_lengths[j]].scatter(-1, all_c[i, j, :all_lengths[j]].unsqueeze(-1), 1)
            # The label position is filled with 0
            mask_results[i] = mask_results[i].scatter(-1, labels.unsqueeze(-1), 0)
        return mask_results

    def get_mul_loss(self, N_mask_results, lm_labels, softmax_logits, label_masks):
        mc_forward_num = softmax_logits.shape[0]
        mask_results = N_mask_results
        n_logits = (self.config.gama * softmax_logits).exp() * mask_results
        n_logits = n_logits.sum(0).sum(-1)

        # get positive samples
        labels = lm_labels.unsqueeze(0).repeat(mc_forward_num, 1, 1).unsqueeze(-1)
        p_logits = torch.gather((- (self.config.gama * softmax_logits)).exp(), -1, labels)
        p_logits = p_logits.sum(0).squeeze(-1)

        loss = torch.log(1 + math.exp(self.config.m * self.config.gama) * n_logits * p_logits) * label_masks
        return loss.sum()

    def get_mse_loss(self, all_log_softmax_logits, lm_labels):
        mc_forward_num = all_log_softmax_logits.shape[0]
        vocab_size = all_log_softmax_logits.shape[-1]
        loss_fct = nn.NLLLoss(ignore_index=0, reduction='sum')

        all_likelihood_loss = None
        for i in range(mc_forward_num):
            log_softmax_logits = all_log_softmax_logits[i]
            likelihood_loss = loss_fct(log_softmax_logits.reshape(-1, vocab_size), lm_labels.view(-1))
            cur_loss = likelihood_loss
            if all_likelihood_loss is None:
                all_likelihood_loss = cur_loss.unsqueeze(0)
            else:
                all_likelihood_loss = torch.cat((all_likelihood_loss, cur_loss.unsqueeze(0)), dim=0)
        all_likelihood_loss = torch.mean(all_likelihood_loss, dim=0)
        return all_likelihood_loss

    def get_mi_loss(self, label_masks, softmax_logits, all_log_softmax_logits):
        regular_loss = softmax_logits * all_log_softmax_logits
        label_masks = label_masks.unsqueeze(0).unsqueeze(-1)
        regular_loss = (regular_loss * label_masks).sum()
        return -regular_loss

    def compute_loss(self, all_logits, N_mask_results, lm_labels):
        log_softmax = nn.LogSoftmax(dim=-1)
        softmax_fct = nn.Softmax(dim=-1)

        label_masks = torch.ones_like(lm_labels)
        label_masks[lm_labels == 0] = 0

        softmax_logits = softmax_fct(all_logits)
        all_log_softmax_logits = log_softmax(all_logits)

        mul_loss = self.get_mul_loss(N_mask_results, lm_labels, softmax_logits, label_masks)
        mse_loss = self.get_mse_loss(all_log_softmax_logits, lm_labels)
        mi_loss = self.get_mi_loss(label_masks, softmax_logits, all_log_softmax_logits)

        loss = mul_loss + mse_loss + mi_loss
        return loss

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.t5_model.config.use_cache
        return_dict = return_dict if return_dict is not None else self.t5_model.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.t5_model.config.num_layers == self.t5_model.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.t5_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.t5_model.model_parallel:
            torch.cuda.set_device(self.t5_model.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self.t5_model._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.t5_model.model_parallel:
            torch.cuda.set_device(self.t5_model.decoder.first_device)
            hidden_states = hidden_states.to(self.t5_model.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.t5_model.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.t5_model.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.t5_model.decoder.first_device)

        # Record the results of using dropout
        decoder_inputs_embeds = self.t5_model.decoder.embed_tokens(decoder_input_ids)
        decoder_inputs_embeds = self.t5_model.decoder.dropout(decoder_inputs_embeds)

        # Close decoder last dropout
        last_dropout = None
        for m in self.t5_model.decoder.modules():
            if m.__class__.__name__.startswith('Dropout'):
                last_dropout = m
        last_dropout.eval()

        # Decode
        decoder_outputs = self.t5_model.decoder(
            # input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Open dropout
        last_dropout.train()
        last_layer_hiddens = decoder_outputs[0]

        all_logits = self.uncertainty_aware_predictions(last_layer_hiddens)
        N_mask_results = self.negative_samples_acquisition(all_logits, labels)
        loss = self.compute_loss(all_logits, N_mask_results, labels)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=all_logits,
        )
