def train_model(
    train_data_path: str,
    model,
    tokenizer,
    output_dir: str = "outputs",
    num_epochs: int = 2,
    lora_r: int = 16,
    max_seq_length: int = 2048,
):
    """
    Fine-tune a language model using QLoRA on a dataset of code snippets with time complexity annotations.

    Args:
        train_data_path (str): Path to the training .jsonl file.
        output_dir (str): Directory where the fine-tuned model will be saved.
        num_epochs (int): Number of training epochs.
        lora_r (int): Rank of the LoRA adaptation layers.
        max_seq_length (int): Maximum input sequence length.

    Returns:
        transformers.TrainingStats: Training statistics and metrics.
    """
    
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from datasets import Dataset
    import json

    # Extract the data from the test_data file
    with open(train_data_path, 'r') as f:
        train_data = [json.loads(line.strip()) for line in f]

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template([convo], tokenize = False, add_generation_prompt = False) for convo in convos]
        texts[1] = texts[1][len("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 July 2024\n\n<|eot_id|>"):]
        return texts[0]+texts[1]

    def gen():
        for code in train_data:
            prompt = f"""Analyze the time complexity of the following code.
        Choose exactly one of the following options: O(1), O(logn), O(n), O(nlogn), O(n^2), O(n^3) or exponential (O(2^n), O(3^n), etc.).
        Give the time complexity of the code:
        {code['src']}"""
            yield {"conversations": [{"role": "user", "content": prompt}, {"role": "assistant", "content": code['complexity']}],
                "text": formatting_prompts_func({"conversations": [{"role": "user", "content": prompt}, {"role": "assistant", "content": code['complexity']}]})}

    dataset = Dataset.from_list(list(gen()))

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = num_epochs,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            report_to = "none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer_stats = trainer.train()
    return trainer_stats