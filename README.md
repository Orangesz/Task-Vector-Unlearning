# 🧩 Project Title
This project aims to compare and improve several unlearning methods. LLM models have a history of sourcing training data through unethical means, and often times this means violating copyrights or intellectual property rights. This also leads to the possibility of sensitive information being used to train models. Unlearning is an important part of rectifying some of these issues, allowing for a stricter focus on ethical and legal sourcing of information while allowing users to opt out of having their data used to train models.

We used several standard methods of unlearning (Gradient Ascent, KL Divergence) and a more experimental method of unlearning (Task Vector Unlearning) and compared the effects of each method on multilingual data sets.

Additionally, an entire write-up on this project is provided as a PDF in this repo.

---

## ⚙️ Setup
```
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage
```
python main.py
```