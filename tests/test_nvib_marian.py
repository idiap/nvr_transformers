#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Set directory one higher
import os.path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import pytorch_lightning as pl
import torch
from transformers import AutoConfig, AutoTokenizer, MarianModel, MarianMTModel

from models_nvib.modeling_nvibmarian import MarianModelNVIB, MarianMTModelNVIB

# The cross and self attention works
# from models_nvib.modeling_nvibmarian_cross_self import (
#     MarianModelNVIB,
#     MarianMTModelNVIB,
# )


# The cross attention works
# from models_nvib.modeling_nvibmarian_cross import MarianModelNVIB, MarianMTModelNVIB


############################################################################
# tst the integration of nvib to the marian transformer translation model
############################################################################
# torch.use_deterministic_algorithms(True)

tolerance = 1e-4

model_path = "Helsinki-NLP/opus-mt-de-en"

tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="de", tgt_lang="en")
config = AutoConfig.from_pretrained(model_path)

config.output_attentions = True

# NVIB config parameters
config.prior_mus_cross = None
config.prior_vars_cross = None
config.prior_log_alphas_cross = None
config.prior_log_alpha_stdevs_cross = None

config.prior_mus_encoder = None
config.prior_vars_encoder = None
config.prior_log_alphas_encoder = None
config.prior_log_alpha_stdevs_encoder = None

config.prior_mus_decoder = None
config.prior_vars_decoder = None
config.prior_log_alphas_decoder = None
config.prior_log_alpha_stdevs_decoder = None

config.delta = 1
config.kappa = 1
config.mu_tau = 1
config.stdev_tau_e = 0  # encoder self attention
config.stdev_tau_c = 0  # cross attention
config.stdev_tau_d = 0  # decoder self attention
config.alpha_tau_e = 10  # encoder self attention
config.alpha_tau_c = 10  # cross attention
config.alpha_tau_d = 10  # decoder self attention
config.dropout = 0.0
config.use_cache = False

marian_model = MarianModel.from_pretrained(
    model_path,
    config=config,
)

nvib_model = MarianModelNVIB.from_pretrained(
    model_path,
    config=config,
)

# Reinitialise ALL the NVIB parameters
nvib_model.nvib_layer.init_parameters()
# Initialise the NVIB parameters per layer
for layer in nvib_model.encoder.layers:
    # if layer has attribute sa_nvib_layer, then init_parameters
    if hasattr(layer, "nvib_sa_layer"):
        layer.nvib_sa_layer.init_parameters()

for layer in nvib_model.decoder.layers:
    # if layer has attribute sa_nvib_layer, then init_parameters
    if hasattr(layer, "nvib_causal_sa_layer"):
        layer.nvib_causal_sa_layer.init_parameters()


# Inputs for the model
inputs = tokenizer(
    [
        "Studien haben gezeigt dass es hilfreich ist einen Hund zu besitzen",
        "Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit, den Wahlbetrug zu bekämpfen.",
        "Eins ist sicher: diese neuen Bestimmungen werden sich negativ auf die Wahlbeteiligung auswirken.",
        "In diesem Sinne untergraben diese Maßnahmen teilweise das demokratische System der USA.",
        "Im Gegensatz zu Kanada sind die US-Bundesstaaten für die Durchführung der Wahlen in den einzelnen Staaten verantwortlich.",
    ],
    padding=True,
    return_tensors="pt",
)

decoder_inputs = tokenizer(
    [
        "Studies have been shown that owning a dog is good for you",
        "Republican leaders justified their policy by the need to combat electoral fraud.",
        "One thing is certain: these new provisions will have a negative impact on voter turn-out.",
        "In this sense, the measures will partially undermine the American democratic system.",
        "Unlike in Canada, the American States are responsible for the organisation of federal elections in the United States.",
    ],
    return_tensors="pt",
    padding=True,
    add_special_tokens=False,
)

#############################################################################################################
# TRAIN MODE
#############################################################################################################


marian_model.train()
nvib_model.train()

# Set seed
pl.seed_everything(42)
marian_outputs = marian_model(
    input_ids=inputs.input_ids,
    decoder_input_ids=decoder_inputs.input_ids,
    attention_mask=inputs.attention_mask,
)
# Set seed
pl.seed_everything(42)
nvib_outputs = nvib_model(
    input_ids=inputs.input_ids,
    decoder_input_ids=decoder_inputs.input_ids,
    attention_mask=inputs.attention_mask,
)


def test_marian_encoder_output_train():
    """Test the encoder output is the same for both models in training mode."""
    # torch.sum(abs(marian_outputs.encoder_last_hidden_state) - abs(nvib_outputs.encoder_last_hidden_state))
    # breakpoint()
    # Encoder output should always be the same assuming you ignore the prior
    assert torch.allclose(
        marian_outputs.encoder_last_hidden_state,
        nvib_outputs.encoder_last_hidden_state[:, 1:, :],
        atol=tolerance,
    )


def test_marian_cross_attention_train():
    """Test the cross attention is the same for both models in training mode."""
    # torch.sum(abs(marian_outputs.cross_attentions[0][0,0,0,:] - nvib_outputs.cross_attentions[0][:,:,:,1:][0,0,0,:]))
    # torch.sum(marian_outputs.cross_attentions[0] -nvib_outputs.cross_attentions[0][:,:,:,1:])
    # The output of the decoder should be the same if initialised correctly
    bool_total = True
    for i in range(0, len(marian_outputs.cross_attentions)):
        booli = torch.allclose(
            marian_outputs.cross_attentions[i],
            nvib_outputs.cross_attentions[i][:, :, :, 1:],
            atol=tolerance,
        )
        bool_total = bool_total and booli
        print("Cross attention layer " + str(i) + " approximation is close " + str(booli))
    assert bool_total


def test_marian_decoder_output_train():
    """Test the decoder output is the same for both models in training mode."""

    # breakpoint()
    # torch.sum(abs(marian_outputs.last_hidden_state) - abs(nvib_outputs.last_hidden_state))
    # The output of the decoder should be the same if initialised correctly
    assert torch.allclose(
        marian_outputs.last_hidden_state,
        nvib_outputs.last_hidden_state,
        atol=1e-3,
    ), "In training mode, there is dropout which makes the output different. Not the end of the world"


# def test_marian_translated_output_training():
#     marian_model = MarianMTModel.from_pretrained(
#         "Helsinki-NLP/opus-mt-en-de",
#         config=config,
#     )
#     nvib_model = MarianMTModelNVIB.from_pretrained(
#         "Helsinki-NLP/opus-mt-en-de",
#         config=config,
#     )

#     # Reinitialise ALL the NVIB parameters
#     nvib_model.model.nvib_layer.init_parameters()
#     # Initialise the NVIB parameters per layer
#     for layer in nvib_model.model.encoder.layers:
#         # if layer has attribute sa_nvib_layer, then init_parameters
#         if hasattr(layer, "nvib_sa_layer"):
#             layer.nvib_sa_layer.init_parameters()

#     for layer in nvib_model.model.decoder.layers:
#         # if layer has attribute sa_nvib_layer, then init_parameters
#         if hasattr(layer, "nvib_causal_sa_layer"):
#             layer.nvib_causal_sa_layer.init_parameters()

#     marian_model.train()
#     nvib_model.train()

#     # Autoregressive prediction
#     pl.seed_everything(42)
#     marian_generated_ids = marian_model.generate(
#         inputs.input_ids,
#         max_new_tokens=50,
#     )
#     marian_predictions = tokenizer.batch_decode(
#         marian_generated_ids, skip_special_tokens=True
#     )

#     pl.seed_everything(42)
#     nvib_generated_ids = nvib_model.generate(
#         inputs.input_ids,
#         max_new_tokens=50,
#     )
#     nvib_predictions = tokenizer.batch_decode(
#         nvib_generated_ids, skip_special_tokens=True
#     )

#     # The output of the decoder should be the same if initialised correctly
#     assert (
#         marian_predictions == nvib_predictions
#     ), "In training mode, the latent layer samples which makes the output different due to the random state. You can set the random state before and after the latent layer to fix this"


#############################################################################################################
# EVAL MODE
#############################################################################################################

marian_model.eval()
nvib_model.eval()

# Set seed
pl.seed_everything(42)
marian_outputs = marian_model(
    input_ids=inputs.input_ids,
    decoder_input_ids=decoder_inputs.input_ids,
    attention_mask=inputs.attention_mask,
)
# Set seed
pl.seed_everything(42)
nvib_outputs = nvib_model(
    input_ids=inputs.input_ids,
    decoder_input_ids=decoder_inputs.input_ids,
    attention_mask=inputs.attention_mask,
)


def test_marian_encoder_output_eval():
    """Test the encoder output is the same for both models in eval mode."""

    # Encoder output should always be the same assuming you ignore the prior
    assert torch.allclose(
        marian_outputs.encoder_last_hidden_state,
        nvib_outputs.encoder_last_hidden_state[:, 1:, :],
        atol=tolerance,
    )


def test_marian_cross_attention_eval():
    """Test the cross attention is the same for both models in eval mode."""

    # breakpoint()
    # torch.sum(abs(marian_outputs.cross_attentions[0][0,0,0,:] - nvib_outputs.cross_attentions[0][:,:,:,1:][0,0,0,:]))

    # The output of the decoder should be the same if initialised correctly
    bool_total = True
    for i in range(0, len(marian_outputs.cross_attentions)):
        booli = torch.allclose(
            marian_outputs.cross_attentions[i],
            nvib_outputs.cross_attentions[i][:, :, :, 1:],
            atol=tolerance,
        )
        bool_total = bool_total and booli
        print("Cross attention layer " + str(i) + " approximation is close " + str(booli))
    assert bool_total


def test_marian_decoder_output_eval():
    """Test the decoder output is the same for both models in eval mode."""

    # breakpoint()
    # The output of the decoder should be the same if initialised correctly
    assert torch.allclose(
        marian_outputs.last_hidden_state, nvib_outputs.last_hidden_state, atol=1e-3
    )


def test_marian_translated_output_eval():
    """Test the autoregressive generation is the same for both models in eval mode."""

    marian_model = MarianMTModel.from_pretrained(
        "Helsinki-NLP/opus-mt-en-de",
        config=config,
    )
    nvib_model = MarianMTModelNVIB.from_pretrained(
        "Helsinki-NLP/opus-mt-en-de",
        config=config,
    )

    # Reinitialise ALL the NVIB parameters
    nvib_model.model.nvib_layer.init_parameters()
    # Initialise the NVIB parameters per layer
    for layer in nvib_model.model.encoder.layers:
        # if layer has attribute sa_nvib_layer, then init_parameters
        if hasattr(layer, "nvib_sa_layer"):
            layer.nvib_sa_layer.init_parameters()

    for layer in nvib_model.model.decoder.layers:
        # if layer has attribute sa_nvib_layer, then init_parameters
        if hasattr(layer, "nvib_causal_sa_layer"):
            layer.nvib_causal_sa_layer.init_parameters()

    marian_model.eval()
    nvib_model.eval()

    # Autoregressive prediction
    pl.seed_everything(42)
    marian_generated_ids = marian_model.generate(
        inputs.input_ids,
        max_new_tokens=50,
    )
    marian_predictions = tokenizer.batch_decode(marian_generated_ids, skip_special_tokens=True)

    pl.seed_everything(42)
    nvib_generated_ids = nvib_model.generate(
        inputs.input_ids,
        max_new_tokens=50,
    )
    nvib_predictions = tokenizer.batch_decode(nvib_generated_ids, skip_special_tokens=True)

    # The output of the decoder should be the same if initialised correctly
    assert marian_predictions == nvib_predictions


# Are training and eval mode the same?
# def tst_nvib_train_vs_eval_encoder_output():

#     assert torch.allclose(
#         nvib_outputs_train.encoder_last_hidden_state,
#         nvib_outputs_eval.encoder_last_hidden_state,
#         atol=tolerance,
#     )


# def tst_nvib_train_vs_eval_decoder_output():

#     assert torch.allclose(
#         nvib_outputs_train.last_hidden_state, nvib_outputs_eval.last_hidden_state, atol=tolerance
#     )


# def tst_nvib_train_vs_eval_cross_attention():

#     #  The output of the decoder should be the same if initialised correctly
#     bool_total = True
#     for i in range(0, len(nvib_outputs_eval.cross_attentions)):
#         booli = torch.allclose(
#             nvib_outputs_eval.cross_attentions[i],
#             nvib_outputs_train.cross_attentions[i],
#             atol=tolerance,
#         )
#         bool_total = bool_total and booli
#         print("Cross attention layer " + str(i) + " approximation is close " + str(booli))
#     assert bool_total


def main():
    """Run the tests."""
    test_marian_encoder_output_train()
    test_marian_cross_attention_train()
    test_marian_decoder_output_train()
    # # test_marian_translated_output_training()
    test_marian_encoder_output_eval()
    test_marian_cross_attention_eval()
    test_marian_decoder_output_eval()

    # This tests the autoregressive generation requires caching
    test_marian_translated_output_eval()


if __name__ == "__main__":
    main()
