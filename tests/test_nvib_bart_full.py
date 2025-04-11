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
from transformers import AutoConfig, AutoTokenizer, BartForConditionalGeneration

from models_hf.modeling_bart import BartModel
from models_nvib.modeling_nvibbart import (
    BartForConditionalGenerationNVIB,
    BartModelNVIB,
)

############################################################################
# Test the integration of nvib to the bart transformer summarisation model
############################################################################
# torch.use_deterministic_algorithms(True)

tolerance = 1e-3

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
config = AutoConfig.from_pretrained("facebook/bart-large-xsum")

config.output_attentions = True

# NVIB config parameters
# config.prior_mu = None
# config.prior_var = None
# config.prior_alpha = None
config.delta = 1
config.kappa = 1
config.mu_tau = 1
config.stdev_tau_e = 0
config.stdev_tau_c = 0
config.stdev_tau_d = 0
config.alpha_tau_e = 10
config.alpha_tau_c = 10
config.alpha_tau_d = 10
config.dropout = 0.0

config.prior_mus_encoder = None
config.prior_vars_encoder = None
config.prior_log_alphas_encoder = None
config.prior_log_alpha_stdevs_encoder = None

# Decoder
config.prior_mus_decoder = None
config.prior_vars_decoder = None
config.prior_log_alphas_decoder = None
config.prior_log_alpha_stdevs_decoder = None

# Cross
config.prior_mus_cross = None
config.prior_vars_cross = None
config.prior_log_alphas_cross = None
config.prior_log_alpha_stdevs_cross = None

# config.decoder_layers = 1
# config.num_beams = 1
# config.use_cache = False

# breakpoint()
bart_model = BartModel.from_pretrained(
    "facebook/bart-large-xsum",
    config=config,
)

nvib_model = BartModelNVIB.from_pretrained(
    "facebook/bart-large-xsum",
    config=config,
)

# Reinitialise ALL the NVIB parameters
if hasattr(nvib_model, "nvib_layer"):
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
        "Here is another example of a short story. I am drunk and silly and I am 80 years old. I have a big family. I like to go skydiving.",
        # "This is a short story of my life. I was born in 1990 and I am 30 years old. I have a dog and a cat. I like to go for walks on the beach.",
        # 'The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed. Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water. Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct. Many businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town. First Minister Nicola Sturgeon visited the area to inspect the damage. The waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare. Jeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit. However, she said more preventative work could have been carried out to ensure the retaining wall did not fail. "It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we are neglected or forgotten," she said. "That may not be true but it is perhaps my perspective over the last few days. "Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?" Meanwhile, a flood alert remains in place across the Borders because of the constant rain. Peebles was badly hit by problems, sparking calls to introduce more defences in the area. Scottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs. The Labour Partys deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand. He said it was important to get the flood protection plan right but backed calls to speed up the process. "I was quite taken aback by the amount of damage that has been done," he said. "Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses." He said it was important that "immediate steps" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans. Have you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.',
        # 'A fire alarm went off at the Holiday Inn in Hope Street at about 04:20 BST on Saturday and guests were asked to leave the hotel. As they gathered outside they saw the two buses, parked side-by-side in the car park, engulfed by flames. One of the tour groups is from Germany, the other from China and Taiwan. It was their first night in Northern Ireland. The driver of one of the buses said many of the passengers had left personal belongings on board and these had been destroyed. Both groups have organised replacement coaches and will begin their tour of the north coast later than they had planned. Police have appealed for information about the attack. Insp David Gibson said: "It appears as though the fire started under one of the buses before spreading to the second. "While the exact cause is still under investigation, it is thought that the fire was started deliberately."',
    ],
    padding=True,
    return_tensors="pt",
)

decoder_inputs = tokenizer(
    [
        "I like to go skydiving.",
        # "I was born in 1990 and have a dog and a cat.",
        # "Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank.",
        # "Two tourist buses have been destroyed by fire in a suspected arson attack in Belfast city centre.",
    ],
    return_tensors="pt",
    padding=True,
    add_special_tokens=False,
)


#############################################################################################################
# TRAIN MODE
#############################################################################################################


bart_model.train()
nvib_model.train()

# Set seed
pl.seed_everything(42)
bart_outputs = bart_model(
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


def test_bart_encoder_output_train():
    """Test the encoder output during training."""
    # torch.sum(abs(bart_outputs.encoder_last_hidden_state) - abs(nvib_outputs.encoder_last_hidden_state))

    # Encoder output should always be the same assuming you ignore the prior
    assert torch.allclose(
        bart_outputs.encoder_last_hidden_state,
        nvib_outputs.encoder_last_hidden_state[:, 1:, :],  # Remove the prior
        # nvib_outputs.encoder_last_hidden_state,
        atol=tolerance,
    )


def test_bart_cross_attention_train():
    """Test the cross attention layers during training."""
    # torch.sum(abs(bart_outputs.cross_attentions[0][0,0,0,:] - nvib_outputs.cross_attentions[0][:,:,:,1:][0,0,0,:]))
    # torch.sum(bart_outputs.cross_attentions[0] -nvib_outputs.cross_attentions[0][:,:,:,1:])
    # The output of the decoder should be the same if initialised correctly
    # bart_outputs.cross_attentions[0][0,0,0,:10]
    # nvib_outputs.cross_attentions[0][0,0,0,:10]

    bool_total = True
    for i in range(0, len(bart_outputs.cross_attentions)):
        booli = torch.allclose(
            bart_outputs.cross_attentions[i],
            nvib_outputs.cross_attentions[i][:, :, :, 1:],
            # nvib_outputs.cross_attentions[i],
            atol=tolerance,
        )
        bool_total = bool_total and booli
        print("Cross attention layer " + str(i) + " approximation is close " + str(booli))
    assert bool_total


def test_bart_decoder_output_train():
    """Test the decoder output during training."""
    # breakpoint()
    # torch.sum(abs(bart_outputs.last_hidden_state) - abs(nvib_outputs.last_hidden_state))
    # The output of the decoder should be the same if initialised correctly
    assert torch.allclose(
        bart_outputs.last_hidden_state, nvib_outputs.last_hidden_state, atol=tolerance
    ), "In training mode, there is dropout which makes the output different. Not the end of the world"


# def test_bart_summarised_output_training():
#     bart_model = BartForConditionalGeneration.from_pretrained(
#         "facebook/bart-large-xsum",
#         config=config,
#     )

#     nvib_model = BartForConditionalGenerationNVIB.from_pretrained(
#         "facebook/bart-large-xsum",
#         config=config,
#     )

#     # Initialise NVIB properly because something overides it...
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

#     bart_model.train()
#     nvib_model.train()

#     # Autoregressive prediction
#     pl.seed_everything(42)
#     bart_generated_ids = bart_model.generate(
#         inputs.input_ids,
#         max_new_tokens=50,
#     )
#     bart_predictions = tokenizer.batch_decode(
#         bart_generated_ids, skip_special_tokens=True
#     )

#     pl.seed_everything(42)
#     nvib_generated_ids = nvib_model.generate(
#         inputs.input_ids,
#         max_new_tokens=50,
#     )
#     nvib_predictions = tokenizer.batch_decode(
#         nvib_generated_ids, skip_special_tokens=True
#     )
#     # breakpoint()
#     # The output of the decoder should be the same if initialised correctly
#     assert (
#         bart_predictions == nvib_predictions
#     ), "In training mode, the latent layer samples which makes the output different due to the random state. You can set the random state before and after the latent layer to fix this"


# #############################################################################################################
# # EVAL MODE
# #############################################################################################################

bart_model.eval()
nvib_model.eval()

# Set seed
pl.seed_everything(42)
bart_outputs = bart_model(
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


def test_bart_encoder_output_eval():
    """Test the encoder output during evaluation."""

    # Encoder output should always be the same assuming you ignore the prior

    assert torch.allclose(
        bart_outputs.encoder_last_hidden_state,
        nvib_outputs.encoder_last_hidden_state[:, 1:, :],
        atol=tolerance,
    )


def test_bart_cross_attention_eval():
    """Test the cross attention layers during evaluation."""

    # torch.sum(abs(bart_outputs.cross_attentions[0][0,0,0,:] - nvib_outputs.cross_attentions[0][:,:,:,1:][0,0,0,:]))

    # The output of the decoder should be the same if initialised correctly
    bool_total = True
    for i in range(0, len(bart_outputs.cross_attentions)):
        booli = torch.allclose(
            bart_outputs.cross_attentions[i],
            nvib_outputs.cross_attentions[i][:, :, :, 1:],
            atol=tolerance,
        )
        bool_total = bool_total and booli
        print("Cross attention layer " + str(i) + " approximation is close " + str(booli))
    assert bool_total


def test_bart_decoder_output_eval():
    """Test the decoder output during evaluation."""

    # breakpoint()
    # The output of the decoder should be the same if initialised correctly
    assert torch.allclose(
        bart_outputs.last_hidden_state, nvib_outputs.last_hidden_state, atol=1e-3
    )


def test_bart_summ_output_eval():
    """Test the summarised output during evaluation."""

    bart_model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large-xsum",
        config=config,
    )

    nvib_model = BartForConditionalGenerationNVIB.from_pretrained(
        "facebook/bart-large-xsum",
        config=config,
    )

    # Initialise NVIB properly because something overides it...
    if hasattr(nvib_model.model, "nvib_layer"):
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

    bart_model.eval()
    nvib_model.eval()

    # Autoregressive prediction
    pl.seed_everything(42)
    bart_generated_ids = bart_model.generate(
        inputs.input_ids,
        max_new_tokens=50,
    )
    bart_predictions = tokenizer.batch_decode(bart_generated_ids, skip_special_tokens=True)

    pl.seed_everything(42)
    nvib_generated_ids = nvib_model.generate(
        inputs.input_ids,
        max_new_tokens=50,
    )
    nvib_predictions = tokenizer.batch_decode(nvib_generated_ids, skip_special_tokens=True)
    # The output of the decoder should be the same if initialised correctly
    # breakpoint()
    assert bart_predictions == nvib_predictions


def main():
    """Run the tests."""
    test_bart_encoder_output_train()
    test_bart_cross_attention_train()
    test_bart_decoder_output_train()

    test_bart_encoder_output_eval()
    test_bart_cross_attention_eval()
    test_bart_decoder_output_eval()
    test_bart_summ_output_eval()

    # test_bart_summarised_output_training()
    pass


if __name__ == "__main__":
    main()
