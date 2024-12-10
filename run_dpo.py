import torch
from trl import SFTTrainer, DPOTrainer, DPOConfig, PPOTrainer, PPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, TextStreamer
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

    return model, tokenizer

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )

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
    trainer.train()
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
def train_ppo(base_model, tokenizer):
    print("Entering PPO...")

    training_args = PPOConfig( 
        output_dir="ppo_output",
    )

    # Enable gradient checkpointing
    base_model.gradient_checkpointing_enable()

    policy = AutoModelForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "nvidia/Llama-3.1-Nemotron-70B-Reward"
    )


    ppo_trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        value_model=None,
        train_dataset=None,
        eval_dataset=None,
        peft_config=None,
    )

    print("Starting PPO training...")
    ppo_trainer.train()
    print("PPO Training completed!")

    # Save the final model
    ppo_trainer.save_model("final_model")
    return ppo_trainer.model

def test_model(model, tokenizer):
    print("\nTesting the model...")
    model = FastLanguageModel.for_inference(model)

    test_messages = [
        {"from": "human", "value": "What is the meaning of life?"},
    ]
    inputs = tokenizer.apply_chat_template(
        test_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    print("Model response:")
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True
    )

if __name__ == "__main__":
    # First run SFT
    sft_model, tokenizer = train_sft()

    # Then run DPO
    # final_model = train_dpo(sft_model, tokenizer)

    final_model = train_ppo(sft_model, tokenizer)

    # Test the final model
    test_model(final_model, tokenizer)
