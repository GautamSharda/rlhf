import torch
from trl import SFTTrainer, DPOTrainer, DPOConfig, PPOTrainer, PPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, TextStreamer, BitsAndBytesConfig
from copy import deepcopy
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import time
import re

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

def clean_generated_text(text):
    """Improved cleaning of generated text."""
    # Remove the prompt from the beginning if it appears
    prompt_pattern = r'^.*?\?(?=\n|$)'
    text = re.sub(prompt_pattern, '', text)
    
    # Remove template tokens
    text = re.sub(r'<\|im_start\|>(?:user|assistant|system)', '', text)
    
    # Remove redundant newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove repeated content
    lines = text.split('\n')
    unique_lines = []
    for line in lines:
        if line not in unique_lines:
            unique_lines.append(line)
    text = '\n'.join(unique_lines)
    
    # Remove any remaining prompt echoes
    text = re.sub(r'^.*?(?:What is|How does|Why|When|Where)\s+.*?\?\s*', '', text)
    
    return text.strip()

def test_model(base, model, tokenizer):
    print("\nTesting the model...")
    test_prompts = [
        "What is the meaning of life?",
        "How does consciousness work?",
        "What is the future of technology?"
    ]
    
    responses_before = []
    responses_after = []
    
    for prompt in test_prompts:
        # Test base model
        base = FastLanguageModel.for_inference(base)
        base.eval()
        test_messages = [{"from": "human", "value": prompt}]
        inputs = tokenizer.apply_chat_template(
            test_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        output = base.generate(
            input_ids=inputs,
            max_new_tokens=200,
            min_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = clean_generated_text(tokenizer.decode(output[0], skip_special_tokens=True))
        responses_before.append(response)
        
        # Test DPO model
        model = FastLanguageModel.for_inference(model)
        model.eval()
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=200,
            min_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = clean_generated_text(tokenizer.decode(output[0], skip_special_tokens=True))
        responses_after.append(response)
    
    # Write results to file with the exact format expected by evaluate_quality.py
    with open("test_results.txt", "w") as file:
        file.write("Responses before DPO feedback loop:\n")
        for prompt, response in zip(test_prompts, responses_before):
            file.write(f"Prompt: {prompt}\n")
            file.write(f"Response: {response}\n\n")

        file.write("Responses after DPO feedback loop:\n")
        for prompt, response in zip(test_prompts, responses_after):
            file.write(f"Prompt: {prompt}\n")
            file.write(f"Response: {response}\n\n")

if __name__ == "__main__":
    # First run SFT
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        base_model,
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

    # Then run DPO
    # final_model = train_dpo(sft_model, tokenizer)

    final_model = train_dpo(model, tokenizer)

    # Test the final model
    test_model(model, final_model, tokenizer)
