import torch
from trl import SFTTrainer, DPOTrainer, DPOConfig, PPOTrainer, PPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, TextStreamer, BitsAndBytesConfig
from copy import deepcopy
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import time

# First part: Supervised Fine-Tuning
def train_sft():
    print("Starting SFT Training...")
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )

    # Apply chat template before returning
    tokenizer = get_chat_template(
        tokenizer,
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        chat_template="chatml",
    )

    def apply_template(examples):
        messages = examples["conversations"]
        text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
        return {"text": text}

    print("Loading SFT dataset...")
    dataset = load_dataset("mlabonne/FineTome-100k", split="train[:1000]")
    dataset = dataset.map(apply_template, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=3e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=5,
            output_dir="sft_output",
            seed=0,
        ),
    )

    print("Starting SFT training...")
    # trainer.train()
    print("SFT Training completed!")
    return model, tokenizer

# Second part: Direct Preference Optimization (will switch to PPO later)
def train_dpo(base_model, tokenizer):
    print("Starting DPO Training...")

    # Load preference dataset
    preference_dataset = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train[:1000]")

    # Create a custom DPOTrainer class to handle logging because the default one throws error when logging
    class CustomDPOTrainer(DPOTrainer):
        def log(self, logs, start_time=None):
            """Override log method to handle both 2 and 3 argument versions"""
            if start_time:
                logs["time"] = time.time() - start_time
            super().log(logs)

    training_args = DPOConfig(
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=1,
        gradient_accumulation_steps=2,
        save_strategy="epoch",
        logging_steps=10,
        output_dir="dpo_output",
        optim="adamw_8bit",
        remove_unused_columns=False,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported()
    )

    # Enable gradient checkpointing
    base_model.gradient_checkpointing_enable()

    dpo_trainer = CustomDPOTrainer(
        model=base_model,
        ref_model=None,  # We'll use implicit reference model
        tokenizer=tokenizer,
        train_dataset=preference_dataset,
        args=training_args,
        beta=0.1,  # KL penalty coefficient
        max_prompt_length=512,
        max_length=1024,
    )

    print("Starting DPO training...")
    dpo_trainer.train()
    print("DPO Training completed!")

    # Save the final model
    dpo_trainer.save_model("final_model")
    return dpo_trainer.model

# Second part: PPO
from typing import Optional

def train_ppo(base_model, tokenizer):
    print("Starting PPO Training...")
    
    # Basic PPO configuration
    ppo_config = PPOConfig(
        output_dir="ppo_output",
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        num_train_epochs=1,
        gradient_accumulation_steps=1,
        save_strategy="epoch",
        logging_steps=10,
        optim="adamw_8bit"
    )

    # Start by setting up all models
    print("Setting up models...")
    policy = base_model  # Don't unwrap PEFT model
    policy.train()  # Ensure in training mode

    # Create reward model
    print("Loading reward model...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "lvwerra/distilbert-imdb",
        num_labels=2,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16
    ).cuda()
    reward_model.eval()
    print("Successfully loaded reward model")

    # Create dataset
    print("\nPreparing dataset...")
    dataset = load_dataset("Dahoas/rm-static", split="train[:1000]")

    print("\nModel check before PPOTrainer:")
    print(f"Policy type: {type(policy)}")
    print(f"Reward model type: {type(reward_model)}")
    
    print("\nInitializing PPO trainer...")
    try:
        # Initialize trainer with minimal components
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            policy=policy,
            ref_policy=None,
            tokenizer=tokenizer,
            train_dataset=dataset,
            reward_model=reward_model,
            # Don't pass value_model - let PPOTrainer handle it internally
        )

        print("Starting PPO training...")
        ppo_trainer.train()
        print("PPO Training completed!")
        return policy

    except Exception as e:
        print(f"\nError during PPO training setup: {e}")
        print("\nFull traceback:")
        import traceback
        print(traceback.format_exc())
        return base_model

def test_model(base, model, tokenizer):
    print("\nTesting the model...")
    model = FastLanguageModel.for_inference(model)
    model.eval()  # Ensure eval mode

    test_messages = [
        {"from": "human", "value": "What is the meaning of life?"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        test_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    print("Base response:")
    text_streamer = TextStreamer(tokenizer)
    _ = base.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=64,  # Reduced to avoid loops
        use_cache=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    print("Model response:")
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=64,  # Reduced to avoid loops
        use_cache=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

if __name__ == "__main__":
    # First run SFT
    sft_model, tokenizer = train_sft()

    # Then run DPO
    # final_model = train_dpo(sft_model, tokenizer)

    final_model = train_ppo(sft_model, tokenizer)

    # Test the final model
    test_model(sft_model, final_model, tokenizer)
