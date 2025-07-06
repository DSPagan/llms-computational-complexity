def run_inference(
    test_data_path: str,
    model,
    tokenizer,
    save_path: str = "outputs/test_results.jsonl",
    max_seq_length: int = 2048,
    max_new_tokens: int = 1024,
):   
    """
    Run inference on a list of code snippets to estimate their time complexity using a pretrained model.

    Args:
        test_data_path (str): Path to the test .jsonl file.
        model: The language model to use for inference.
        tokenizer: The tokenizer corresponding to the model.
        save_path (str): File path where the model's predictions will be saved (JSONL format).
        max_seq_length (int): Maximum length of the input tokens.
        max_new_tokens (int): Maximum number of tokens to generate for the output.

    Saves:
        A .jsonl file with one entry per code snippet, including:
            - "src": the code snippet
            - "complexity": the true complexity label
            - "model": the model's prediction
    """

    import json

    # Extract the data from the test_data file
    with open(test_data_path, 'r') as f:
        test_data = [json.loads(line.strip()) for line in f]

    with open(save_path, 'w') as f:
        for code in test_data:
            # Change the prompt to match the training data format
            prompt = f"""Analyze the time complexity of the following code.
        Choose exactly one of the following options: O(1), O(logn), O(n), O(nlogn), O(n^2), O(n^3) or exponential (O(2^n), O(3^n), etc.).
        Give the time complexity of the code:
        {code['src']}"""
            messages = [
                {"role": "user", "content": prompt}
                ]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")

            if len(inputs[0]) > max_seq_length:
                continue

            tokens = model.generate(input_ids=inputs,
                                    do_sample=False, # deterministic generation
                                    max_new_tokens=max_new_tokens, # maximum number of tokens to generate
                                    use_cache=True, # use cache for faster generation
                                    no_repeat_ngram_size=4) # to avoid repetition

            result = tokenizer.decode(tokens[0],skip_special_tokens=True)
            idx = result.find("assistant")
            result = result[idx + len("assistant"):].lstrip() # Filter the output to get the assistant's response

            entry = {
                "src": code['src'],
                "complexity": code['complexity'],
                "model": result,
            }
            f.write(json.dumps(entry) + '\n')