import torch
import transformers
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


nltk.download('punkt')

print("All packages are successfully installed and imported.")

# Load the dataset
data_path = r'C:\IHF_RL\amazon.csv.zip'
df = pd.read_csv(data_path)

# Initialize the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Fix padding token warning
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to generate text with controlled sampling
def generate_text(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = inputs.shape[1]
    max_input_length = 1024

    # If input is too long, truncate
    if input_length > max_input_length:
        inputs = inputs[:, -max_input_length:]

    attention_mask = torch.ones(inputs.shape, device=device)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Reward function using BLEU and ROUGE-L
def reward_function(generated_text, reference_text):
    smoothing_function = SmoothingFunction()

    # Calculate BLEU score
    bleu_score = sentence_bleu(
        [reference_text.split()],
        generated_text.split(),
        smoothing_function=smoothing_function.method1
    )

    # Calculate ROUGE-L score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    rouge_l_score = scores['rougeL'].fmeasure

    # Combined reward
    combined_score = 0.5 * bleu_score + 0.5 * rouge_l_score
    return bleu_score, rouge_l_score, combined_score

# Iterative Feedback Loop
def iterative_feedback(product_name, description, reference_text, iterations=5):
    results = []

    for i in range(iterations):
        prompt = (
            "Example of a good product description:\n"
            "'The XYZ USB Cable offers fast charging and data transfer capabilities, compatible with various devices. "
            "Its durable design ensures longevity, while customer support is always available.'\n\n"
            f"Generate a concise product description for:\n"
            f"Product Name: {product_name}\n"
            f"Key Features: {description}\n"
            "Focus on compatibility, charging speed, durability, security, warranty, and a catchy ending."
        )

        generated_text = generate_text(prompt)

        print(f"\nIteration {i + 1}:")
        print("Generated Text:")
        print("--------------------------------------------------")
        print(generated_text.strip())

        # Compute scores
        bleu_score, rouge_l_score, reward = reward_function(generated_text, reference_text)
        print(f"Automatic Reward (BLEU + ROUGE-L): {reward:.4f}")

        # Manual feedback
        try:
            manual_score = float(input("Rate the generated text on a scale of 1 to 10 (higher is better): "))
        except:
            manual_score = 5.0  # default if user skips
            print("No manual score entered, defaulting to 5.0")

        print(f"Manual Feedback Score: {manual_score:.4f}\n")

        # Save results
        results.append({
            'Iteration': i + 1,
            'BLEU Score': bleu_score,
            'ROUGE-L Score': rouge_l_score,
            'Manual Score': manual_score,
            'Combined Reward': reward
        })

        combined_reward = (0.7 * manual_score / 10) + (0.3 * reward)
        print(f"Combined Reward (Manual + Automatic): {combined_reward:.4f}\n")
        print("Adjusting model based on feedback (placeholder)\n")
        print("-" * 50)

    # Plot results
    results_df = pd.DataFrame(results)
    print("\nComparison Table of Scores:")
    print(results_df)

    # Plot metrics
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Iteration'], results_df['BLEU Score'], marker='o', color='green', label='BLEU Score')
    plt.plot(results_df['Iteration'], results_df['ROUGE-L Score'], marker='x', color='orange', label='ROUGE-L Score')
    plt.plot(results_df['Iteration'], results_df['Manual Score'] / 10, marker='s', color='purple', label='Manual Score (Scaled)')
    plt.plot(results_df['Iteration'], results_df['Combined Reward'], marker='^', color='blue', label='Combined Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title(f"Progression of Scores Across Iterations for {product_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Iteration'], results_df['Combined Reward'], marker='o', color='b', label='Combined Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Combined Reward Score')
    plt.title(f"Progression of Combined Reward Score for {product_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage: Process one product from the dataset
for index, row in df.iterrows():
    product_name = row['product_name']
    description = row['about_product']
    reference_text = row['review_content']

    print(f"\nProcessing product: {product_name}\n")
    iterative_feedback(product_name, description, reference_text)
    break  # Remove this break to process the full dataset
