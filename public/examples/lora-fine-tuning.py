import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# -------------------------------------------------------------------
# 1. Create the Dataset using Dataset.from_dict()
# -------------------------------------------------------------------
# We create a small dictionary of standard questions and pirate answers.
pirate_data = {
    "instruction": [
        "Hello, how are you today?",
        "What is your name?",
        "Where is the nearest grocery store?",
        "Can you explain what a computer is?",
        "What is your favorite food?"
    ],
    "response": [
        "Ahoy matey! I be doin' mighty fine, ready to sail the high seas!",
        "They call me Cap'n AI, the most fearsome language model of the Caribbean!",
        "Ye best be checkin' yer treasure map, or I'll make ye walk the plank!",
        "Tis a magical box of shiny lights, fueled by dark sorcery and electricity, arrr!",
        "Nothin' beats a barrel of salted pork and hardtack, savvy?"
    ]
}

# Convert the dictionary into a Hugging Face Dataset
dataset = Dataset.from_dict(pirate_data)

# -------------------------------------------------------------------
# 2. Load the Tokenizer and Model
# -------------------------------------------------------------------
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer and set the padding token
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", # Automatically puts the model on GPU if available
    torch_dtype=torch.bfloat16 # Uses less memory
)

# -------------------------------------------------------------------
# 3. Format the Prompts
# -------------------------------------------------------------------
# SFTTrainer needs to know how to format the data into a single string.
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        # Simple Chat format: User asks, Assistant answers
        text = f"User: {example['instruction'][i]}\nAssistant: {example['response'][i]}"
        output_texts.append(text)
    return output_texts

# -------------------------------------------------------------------
# 4. Configure LoRA (Low-Rank Adaptation)
# -------------------------------------------------------------------
# Instead of training all 1.1 Billion parameters, we only train a tiny 
# fraction of them by injecting small matrices into the attention layers.
peft_config = LoraConfig(
    r=8,                       # Rank of the LoRA matrices (smaller = faster, larger = more capacity)
    lora_alpha=16,             # Scaling factor
    target_modules=["q_proj", "v_proj"], # Which layers to target in the transformer
    lora_dropout=0.05,         # Dropout probability for regularization
    bias="none",
    task_type="CAUSAL_LM"
)

# -------------------------------------------------------------------
# 5. Set up the Trainer and Train!
# -------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./pirate_assistant",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=1,
    max_steps=15,              # Just 15 steps for this quick demonstration
    save_steps=15,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=training_args,
)

print("Hoisting the colors and starting training...")
trainer.train()

# Save the fine-tuned LoRA adapter
trainer.model.save_pretrained("pirate_lora_adapter")
print("Arrr! Training be complete. Adapter saved!")