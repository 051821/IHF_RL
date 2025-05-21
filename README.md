# 🧠 Iterative Feedback Product Description Generator

This project aims to generate high-quality product descriptions using **GPT-2**, refined iteratively through **manual and automatic feedback loops**. It combines **BLEU** and **ROUGE-L** scoring with **human ratings** to reward and improve the generation quality over multiple iterations.

---

## 🚀 Features

- ✅ Uses pre-trained **GPT-2** for product description generation.
- ✅ Calculates **BLEU** and **ROUGE-L** scores for automatic evaluation.
- ✅ Accepts **manual user ratings** for iterative feedback and reward computation.
- ✅ Plots performance metrics (BLEU, ROUGE-L, Manual Scores, Combined Reward) over iterations.
- ✅ Easily adaptable for **product review datasets** or any other text generation use case.

---

## 📁 Dataset

- The input dataset should be in `.csv` format (e.g., `amazon.csv.zip`) containing:
  - `product_name`: Name of the product.
  - `about_product`: Key features or short description.
  - `review_content`: Reference text used for evaluating generated content.

---

## 🛠️ Installation

1. **Clone the repository** (if applicable):

   ```bash
   git clone https://github.com/yourusername/feedback-product-generator.git
   cd feedback-product-generator
Install required packages:

bash
Copy
Edit
pip install torch transformers nltk pandas matplotlib rouge-score
Download NLTK packages (automatically done in code, but manually if needed):

python
Copy
Edit
import nltk
nltk.download('punkt')
💻 Usage
Place your dataset file (e.g., amazon.csv.zip) at the correct location.

Run the script:

bash
Copy
Edit
python main.py
The script will:

Load one product at a time.

Generate a product description using GPT-2.

Ask for manual rating input (1–10).

Combine it with BLEU & ROUGE-L for a combined reward score.

Repeat for multiple iterations.

Plot performance and show a comparative table of all scores.

💡 Tip: You can remove the break statement inside the for loop to process the entire dataset.

📊 Example Output
Example prompt:

yaml
Copy
Edit
Generate a concise product description for:
Product Name: UltraFast USB-C Cable
Key Features: Fast charging, braided design, universal compatibility, 1.5m length
Output description:

csharp
Copy
Edit
The UltraFast USB-C Cable offers high-speed charging and data transfer with a durable braided design. Compatible with phones, tablets, and laptops, this 1.5m cable ensures reliability and longevity with every use.
📈 Evaluation
Automatic Scores:

BLEU Score: Measures n-gram overlap.

ROUGE-L Score: Measures longest common subsequence.

Manual Score:

User input (1–10), scaled and combined with auto scores.

Combined Reward:

ini
Copy
Edit
Combined = 0.7 × (Manual / 10) + 0.3 × (BLEU + ROUGE-L)/2
🧪 Future Improvements
Fine-tune GPT-2 with Reinforcement Learning from Human Feedback (RLHF).

Store feedback history and train a reward model.

Automate manual score using simulated metrics.

Build a Streamlit dashboard or web app for real-time testing.