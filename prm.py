from datasets import load_dataset
import jax
from tunix.models.gemma import model as gemma_lib
from flax import nnx
import jax.numpy as jnp
from orbax import checkpoint as ocp
import qwix
import os
from transformers import AutoTokenizer
import optax
import numpy as np
from tqdm import tqdm

def convert_token_to_id(token, tokenizer):
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    return token_ids[0]

def get_gemma_ref_model(ckpt_path, MESH):
  devices = jax.devices()
  mesh_shape, mesh_axes = MESH
  mesh = jax.make_mesh(devices.reshape(*mesh_shape), mesh_axes)
  model_config = gemma_lib.ModelConfig.gemma2_2b()
  abs_gemma: nnx.Module = nnx.eval_shape(
      lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
  )
  abs_state = nnx.state(abs_gemma)
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(ckpt_path, target=abs_state)

  graph_def, _ = nnx.split(abs_gemma)
  gemma = nnx.merge(graph_def, restored_params)
  return gemma, mesh, model_config


def get_lora_model(base_model, mesh, RANK, ALPHA):
  lora_provider = qwix.LoraProvider(
      module_path=(
          ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
          ".*attn_vec_einsum"
      ),
      rank=RANK,
      alpha=ALPHA,
  )

  model_input = base_model.get_model_input()
  lora_model, lora_params = qwix.apply_lora_to_model(
    base_model, lora_provider, **model_input
  )

  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model, lora_params


def process_data(
      dataset,
      tokenizer,
      placeholder_token,
      max_length
):
    placeholder_token_id = convert_token_to_id(placeholder_token, tokenizer)

    inputs = dataset['input']
    labels = dataset['values']

    processed_data = []

    for i in range(len(inputs)):
        input_text = inputs[i]
        label_values = labels[i]

        encoded = tokenizer(
            input_text,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors="np",
            add_special_tokens=False,
        )
        
        input_ids = jnp.array(encoded["input_ids"][0])
        attention_masks = jnp.array(encoded["attention_mask"][0])

        label_tokens = []
        for label in label_values:
            label_tokens.append(convert_token_to_id(label, tokenizer))
        
        label_tensors = jnp.array(label_tokens, dtype=input_ids.dtype)

        mask = input_ids == placeholder_token_id
        num_placeholders = mask.sum()

        truncated_labels = label_tensors[:num_placeholders]

        full_labels = jnp.full_like(input_ids, -100, dtype=label_tensors.dtype)
        full_labels = full_labels.at[mask].set(truncated_labels)

        processed_data.append((
            input_ids,
            attention_masks,
            full_labels
        ))
    
    return processed_data

def padded_data(batch, pad_token_id):
    input_ids = []
    attention_masks = []
    labels = []

    for x, y, z in batch:
      input_ids.append(x)
      attention_masks.append(y)
      labels.append(z)
    
    max_len = max(len(ids) for ids in input_ids)

    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []

    for ids, attn_mask, lbls in zip(input_ids, attention_masks, labels):
       pad_len = max_len - len(ids)
       
       ids_jax = jnp.array(ids) if not isinstance(ids, jnp.ndarray) else ids
       mask_jax = jnp.array(attn_mask) if not isinstance(attn_mask, jnp.ndarray) else attn_mask
       lbls_jax = jnp.array(lbls) if not isinstance(lbls, jnp.ndarray) else lbls

       padded_input_ids.append(jnp.concatenate([ids_jax, jnp.full((pad_len,), pad_token_id, dtype=ids_jax.dtype)]))
       padded_attention_masks.append(jnp.concatenate([mask_jax, jnp.zeros(pad_len, dtype=mask_jax.dtype)]))
       padded_labels.append(jnp.concatenate([lbls_jax, jnp.full((pad_len,), pad_token_id, dtype=lbls_jax.dtype)]))
    
    return (
        jnp.stack(padded_input_ids),
        jnp.stack(padded_attention_masks),
        jnp.stack(padded_labels),
    )

def prm_loss_fn(logits, input_ids, labels, placeholder_token_id, reward_token_ids):
    placeholder_mask = input_ids == placeholder_token_id
    logits_at_placeholders = logits[placeholder_mask]  # [num_placeholders, vocab_size]
    labels_at_placeholders = labels[placeholder_mask]  # [num_placeholders]

    logits_reward = logits_at_placeholders[:, reward_token_ids]
    label_indices = jnp.zeros_like(labels_at_placeholders, dtype=jnp.int32)
    for i, token_id in enumerate(reward_token_ids):
        label_indices = jnp.where(labels_at_placeholders == token_id, i, label_indices)

    labels_at_placeholders = label_indices.astype(jnp.int32) 
    valid_mask = labels_at_placeholders != -100

    if valid_mask.sum() == 0: return jnp.array(0.0)

    valid_logits = logits_reward[valid_mask]
    valid_labels = labels_at_placeholders[valid_mask]
    
    loss = optax.softmax_cross_entropy_with_integer_labels(valid_logits, valid_labels).mean()
    
    return loss

def filter_lora_grads(grads):
    if isinstance(grads, dict):
        filtered = {}
        for key, value in grads.items():
            if "lora" in key.lower():
                filtered[key] = value
            else:
                filtered[key] = jnp.zeros_like(value)
        return filtered
    else:
        def mask_fn(path, value):
            path_str = "/".join(str(p) for p in path) if isinstance(path, (list, tuple)) else str(path)
            if "lora" in path_str.lower():
                return value
            else:
                return jnp.zeros_like(value)
        
        return jax.tree_util.tree_map_with_path(mask_fn, grads)

def train_step(
    batch, 
    params, 
    model, 
    rngs, 
    placeholder_token_id,
    reward_token_ids
):
    input_ids = jnp.array(batch["input_ids"])
    attention_mask = jnp.array(batch["attention_mask"])
    labels = jnp.array(batch["labels"])

    def loss_fn(params):
      # For nnx models, use model directly instead of model.apply()
      # Note: Adjust this based on your nnx model API
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        rngs=rngs 
      )
      logits = outputs.logits

      loss = prm_loss_fn(
        logits=logits,
        input_ids=input_ids,
        labels=labels,
        placeholder_token_id=placeholder_token_id,
        reward_token_ids=reward_token_ids
      )

      return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)

    filtered_grads = filter_lora_grads(grads)

    return loss, filtered_grads

def train_prm (
    dataset,
    tokenizer,
    placeholder_token, 
    max_length,
    learning_rate,
    lora_params,
    num_epochs,
    batch_size,
    gradient_accumulation_steps,
    warmup_ratio,
    lora_model,
    reward_tokens
):
   
    train_data = process_data(
        dataset,
        tokenizer,
        placeholder_token, 
        max_length
    )

    placeholder_token_id = convert_token_to_id(placeholder_token, tokenizer)
    reward_token_ids = [convert_token_to_id(token, tokenizer) for token in reward_tokens]

    num_batches = len(train_data)//batch_size
    num_update_steps_per_epoch = num_batches // gradient_accumulation_steps
    total_steps = num_update_steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    #creating optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps-warmup_steps,
        end_value=learning_rate*0.1
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b1=0.9, b2=0.95),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)
    )

    opt_state = optimizer.init(lora_params)

    rng = jax.random.PRNGKey(0)
    dropout_rng = jax.random.fold_in(rng, 0)

    step = 0
    global_step = 0
    loss_sum = 0.0
    accumulated_grads = None
    params = lora_params

    #Training Started
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        np.random.shuffle(train_data)

        epoch_bar = tqdm(
            range(num_batches),
            desc=f"Train Step of epoch {epoch + 1}"
        )

        for batch_idx in epoch_bar:
            batch = padded_data(
                train_data[batch_idx:batch_idx+batch_size],
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            )

            input_batch_ids, attention_batch_masks, labels_batch = batch

            step_rng = jax.random.fold_in(dropout_rng, step)
            rngs = {
            'dropout': step_rng
            }

            loss, grads = train_step(
                {
                    "input_ids": input_batch_ids,
                    "attention_mask": attention_batch_masks,
                    "labels": labels_batch,
                },
                params=params,
                model=lora_model,
                rngs=rngs,
                placeholder_token_id=placeholder_token_id,
                reward_token_ids=reward_token_ids
            )

            if accumulated_grads is None:
                    accumulated_grads = grads
            else:
                accumulated_grads = jax.tree.map(lambda a, b: a + b, accumulated_grads, grads)

            loss_sum += float(loss)
            step += 1

            if step % gradient_accumulation_steps == 0:
                global_step = step // gradient_accumulation_steps
                updates, opt_state = optimizer.update(accumulated_grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                avg_loss = loss_sum / gradient_accumulation_steps

                accumulated_grads = jax.tree.map(lambda x: jnp.zeros_like(x), accumulated_grads)
                loss_sum = 0.0

                current_lr = schedule(global_step)
                logs_dict = {
                    "prm_loss": avg_loss,
                    "lr": current_lr,
                }
                epoch_bar.set_postfix(logs_dict)
                print(f"\nStep {global_step}: Loss={avg_loss:.4f}, LR={current_lr:.2e}")

        epoch_bar.close()
    
    print("\nSaving final checkpoint...")
    save_path: str = "./prm_checkpoints"
    final_checkpoint_path = os.path.join(save_path, "final")
    os.makedirs(final_checkpoint_path, exist_ok=True)
    
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(final_checkpoint_path, params)
    tokenizer.save_pretrained(final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")
    
    print("\n=== Training Complete ===")
    return lora_model, params


def main():
    #load dataset
    dataset_name = "zhuzilin/Math-Shepherd"
    train_data = load_dataset(dataset_name, split='train')
    # eval_data = load_dataset(dataset_name, split='validation')
    print(train_data)

    #load model
    ref_model, mesh, model_config = get_gemma_ref_model(
       ckpt_path=os.path.join(os.path.abspath("./intermediate_ckpt/"), "state"),
       MESH = [(1, 4), ("fsdp", "tp")]
    )

    lora_model, lora_params = get_lora_model(
       base_model=ref_model, 
       mesh=mesh,
       RANK = 64,
       ALPHA = 64.0
    )

    #load tokenizer
    MODEL_NAME = "google/gemma-2/flax/gemma2-2b-it" 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token

    train_prm(
        dataset=train_data,
        tokenizer=tokenizer,
        placeholder_token="ки",
        max_length=512,
        learning_rate=1e-5,
        lora_params=lora_params,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        lora_model=lora_model,
        reward_tokens=["+", "-"]
    )
    

if __name__ == "__main__":
    main()