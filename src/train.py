from trl import SFTTrainer
from transformers import TrainingArguments, TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template([convo], tokenize = False, add_generation_prompt = False) for convo in convos]
    texts[1] = texts[1][len("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 July 2024\n\n<|eot_id|>"):]
    return texts[0]+texts[1]

def gen():
  for code in train:
    prompt = f"""Analyze the time complexity of the following code.
                Choose exactly one of the following options: O(1), O(logn), O(n), O(nlogn), O(n^2), O(n^3) or exponential (O(2^n), O(3^n), etc.).
                Give the time complexity of the code:
                {code['src']}"""
    yield {"conversations": [{"role": "user", "content": prompt}, {"role": "assistant", "content": code['complexity']}],
           "text": formatting_prompts_func({"conversations": [{"role": "user", "content": prompt}, {"role": "assistant", "content": code['complexity']}]})}

dataset = Dataset.from_list(list(gen()))

# Add LoRA adapters to the base model
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# steps = ceil((data_length * num_train_epochs) / (per_device_train_batch_size * gradient_accumulation_steps))
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 2, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

trainer_stats = trainer.train()