import os

import torch

from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_name = "othello-gpt"
model = HookedSAETransformer.from_pretrained(model_name)

dataset_path = "taufeeque/othellogpt"
context_size = 59

layer = 5
training_tokens = int(1e8)
train_batch_size_tokens = 2048
n_steps = int(training_tokens / train_batch_size_tokens)


runner_cfg = LanguageModelSAERunnerConfig(
    #
    # Data generation
    model_name=model_name,
    hook_name=f"blocks.{layer}.mlp.hook_post",
    hook_layer=layer,
    d_in=model.cfg.d_mlp,
    dataset_path=dataset_path,
    is_dataset_tokenized=True,
    prepend_bos=False,
    streaming=True,
    train_batch_size_tokens=train_batch_size_tokens,
    context_size=context_size,
    start_pos_offset=5,
    end_pos_offset=5,
    #
    # SAE achitecture
    architecture="gated",
    expansion_factor=8,
    b_dec_init_method="zeros",
    apply_b_dec_to_input=True,
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=True,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    #
    # Activations store
    n_batches_in_buffer=32,
    store_batch_size_prompts=16,
    training_tokens=training_tokens,
    #
    # Training hyperparameters (standard)
    lr=2e-4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    lr_scheduler_name="constant",
    lr_warm_up_steps=int(0.2 * n_steps),
    lr_decay_steps=int(0.2 * n_steps),
    #
    # Training hyperparameters (SAE-specific)
    l1_coefficient=5,
    l1_warm_up_steps=int(0.2 * n_steps),
    use_ghost_grads=False,
    feature_sampling_window=1000,
    dead_feature_window=500,
    dead_feature_threshold=1e-5,
    #
    # Logging / evals
    log_to_wandb=True,
    wandb_project=f"othello_gpt_sae_{layer=}",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=10,
    checkpoint_path="checkpoints",
    #
    # Misc.
    device=str(device),
    seed=42,
    n_checkpoints=5,
    dtype="float32",
)


# t.set_grad_enabled(True)
runner = SAETrainingRunner(runner_cfg)
sae = runner.run()

hf_repo_id = "callummcdougall/arena-demos-othellogpt"
sae_id = "blocks.5.mlp.hook_post-v1"

upload_saes_to_huggingface({sae_id: sae}, hf_repo_id=hf_repo_id)

othellogpt_sae = SAE.from_pretrained(
    release=hf_repo_id, sae_id=sae_id, device=str(device)
)[0]




# # Create vis
# from sae_vis import SaeVisConfig, SaeVisData, SaeVisLayoutConfig

# def load_othello_vocab():
#     all_squares = [r + c for r in "ABCDEFGH" for c in "01234567"]
#     legal_squares = [sq for sq in all_squares if sq not in ["D3", "D4", "E3", "E4"]]
#     # Model's vocabulary = all legal squares (plus "pass")
#     vocab_dict = {
#         token_id: str_token
#         for token_id, str_token in enumerate(["pass"] + legal_squares)
#     }
#     # Probe vocabulary = all squares on the board
#     vocab_dict_probes = {
#         token_id: str_token for token_id, str_token in enumerate(all_squares)
#     }
#     return {
#         "embed": vocab_dict,
#         "unembed": vocab_dict,
#         "probes": vocab_dict_probes,
#     }

# sae_vis_data = SaeVisData.create(
#     sae=othellogpt_sae,
#     model=othellogpt,
#     linear_probes_input=linear_probes,
#     tokens=tokens[:5000],
#     target_logits=target_logits[:5000],
#     cfg=SaeVisConfig(
#         hook_point=othellogpt_sae.cfg.hook_name,
#         features=alive_feats[:256],
#         seqpos_slice=(5, -5),
#         feature_centric_layout=SaeVisLayoutConfig.default_othello_layout(boards=True),
#     ),
#     vocab_dict=load_othello_vocab(),
#     verbose=True,
# )

# sae_vis_data.save_feature_centric_vis(
#     filename=str(section_dir / "feature_vis_othello.html"),
#     verbose=True,
# )

# display_vis_inline(section_dir / "feature_vis_othello.html", height=1400)


# # In[3]:


# total_training_steps = 3_000  # probably we should do more
# batch_size = 4096
# total_training_tokens = total_training_steps * batch_size

# lr_warm_up_steps = 500
# lr_decay_steps = total_training_steps // 5  # 20% of training
# l1_warm_up_steps = total_training_steps // 20  # 5% of training

# cfg = LanguageModelSAERunnerConfig(
#     # Data Generating Function (Model + Training Distibuion)
#     model_name="othello-gpt",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
#     hook_name="blocks.6.hook_resid_pre",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
#     hook_layer=6,  # Only one layer in the model.
#     d_in=512,  # the width of the mlp output.
#     dataset_path='taufeeque/othellogpt',  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
#     is_dataset_tokenized=True,
#     streaming=False,  # we could pre-download the token dataset if it was small.
#     # SAE Parameters
#     mse_loss_normalization=None,  # We won't normalize the mse loss,
#     expansion_factor=16,  # the width of the SAE. Larger will result in better stats but slower training.
#     b_dec_init_method="geometric_median",  # The geometric median can be used to initialize the decoder weights.
#     apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
#     normalize_sae_decoder=False,
#     scale_sparsity_penalty_by_decoder_norm=True,
#     decoder_heuristic_init=True,
#     init_encoder_as_decoder_transpose=True,
#     normalize_activations="expected_average_only_in",
#     # Training Parameters
#     lr=0.00003,  # lower the better, we'll go fairly high to speed up the tutorial.
#     adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
#     adam_beta2=0.999,
#     lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
#     lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
#     lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
#     l1_coefficient=0.001,  # will control how sparse the feature activations are
#     l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
#     lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
#     train_batch_size_tokens=batch_size,
#     context_size=59,  # will control the length of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
#     # Activation Store Parameters
#     n_batches_in_buffer=32,  # controls how many activations we store / shuffle.
#     training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
#     store_batch_size_prompts=32,
#     # Resampling protocol
#     use_ghost_grads=False,  # we don't use ghost grads anymore.
#     feature_sampling_window=500,  # this controls our reporting of feature sparsity stats
#     dead_feature_window=1e6,  # would effect resampling or ghost grads if we were using it.
#     dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
#     # WANDB
#     log_to_wandb=False,  # always use wandb unless you are just testing code.
#     wandb_project="othello_gpt_sae",
#     wandb_log_frequency=30,
#     eval_every_n_wandb_logs=20,
#     # Misc
#     device=device,
#     seed=42,
#     n_checkpoints=0,
#     checkpoint_path="checkpoints",
#     dtype="torch.float32",
#     start_pos_offset=5,
#     end_pos_offset=5
# )
# test = 1
# # look at the next cell to see some instruction for what to do while this is running.
# sparse_autoencoder = SAETrainingRunner(cfg).run()


# # In[9]:


# sparse_autoencoder.W_dec.shape


# # In[4]:


# total_training_steps = 3_000  # probably we should do more
# batch_size = 4096
# total_training_tokens = total_training_steps * batch_size

# lr_warm_up_steps = 0
# lr_decay_steps = total_training_steps // 5  # 20% of training
# l1_warm_up_steps = total_training_steps // 20  # 5% of training

# cfg = LanguageModelSAERunnerConfig(
#     # Data Generating Function (Model + Training Distibuion)
#     model_name="tiny-stories-1L-21M",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
#     hook_name="blocks.0.hook_mlp_out",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
#     hook_layer=0,  # Only one layer in the model.
#     d_in=1024,  # the width of the mlp output.
#     dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
#     is_dataset_tokenized=True,
#     streaming=True,  # we could pre-download the token dataset if it was small.
#     # SAE Parameters
#     mse_loss_normalization=None,  # We won't normalize the mse loss,
#     expansion_factor=16,  # the width of the SAE. Larger will result in better stats but slower training.
#     b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
#     apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
#     normalize_sae_decoder=False,
#     scale_sparsity_penalty_by_decoder_norm=True,
#     decoder_heuristic_init=True,
#     init_encoder_as_decoder_transpose=True,
#     normalize_activations="expected_average_only_in",
#     # Training Parameters
#     lr=5e-5,  # lower the better, we'll go fairly high to speed up the tutorial.
#     adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
#     adam_beta2=0.999,
#     lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
#     lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
#     lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
#     l1_coefficient=5,  # will control how sparse the feature activations are
#     l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
#     lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
#     train_batch_size_tokens=batch_size,
#     context_size=512,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
#     # Activation Store Parameters
#     n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
#     training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
#     store_batch_size_prompts=16,
#     # Resampling protocol
#     use_ghost_grads=False,  # we don't use ghost grads anymore.
#     feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
#     dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
#     dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
#     # WANDB
#     log_to_wandb=False,  # always use wandb unless you are just testing code.
#     wandb_project="sae_lens_tutorial",
#     wandb_log_frequency=30,
#     eval_every_n_wandb_logs=20,
#     # Misc
#     device=device,
#     seed=42,
#     n_checkpoints=0,
#     checkpoint_path="checkpoints",
#     dtype="float32"
# )
# # look at the next cell to see some instruction for what to do while this is running.
# sparse_autoencoder = SAETrainingRunner(cfg).run()


# # # TO DO: Understanding TinyStories-1L with our SAE
# # 
# # I haven't had time yet to complete this section, but I'd love to see a PR where someones uses an SAE they trained in this tutorial to understand this model better.

# # In[ ]:


# import pandas as pd

# # Let's start by getting the top 10 logits for each feature
# projection_onto_unembed = sparse_autoencoder.W_dec @ model.W_U


# # get the top 10 logits.
# vals, inds = torch.topk(projection_onto_unembed, 10, dim=1)

# # get 10 random features
# random_indices = torch.randint(0, projection_onto_unembed.shape[0], (10,))

# # Show the top 10 logits promoted by those features
# top_10_logits_df = pd.DataFrame(
#     [model.to_str_tokens(i) for i in inds[random_indices]],
#     index=random_indices.tolist(),
# ).T
# top_10_logits_df

