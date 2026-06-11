# Smart-Medical-Question-Answering-System-using-LoRA

## Overview

The Smart Medical Question Answering System is an AI-powered healthcare assistant that answers medical questions using a Large Language Model (LLM) fine-tuned with LoRA (Low-Rank Adaptation). Instead of training an entire model from scratch, this project efficiently adapts a pretrained TinyLlama model by training only a small number of additional parameters.

The system can understand medical questions related to diseases, symptoms, treatments, and healthcare information, and generate relevant responses while significantly reducing computational cost and memory requirements.

---

## Features

* Medical Question Answering using Large Language Models
* Parameter-Efficient Fine-Tuning with LoRA
* TinyLlama-based Language Model
* Custom Medical Dataset Support
* Interactive Question-Answer Interface
* Reduced Training Time and Memory Usage
* Lightweight LoRA Adapter Storage
* Easy Deployment and Inference

---

## Project Workflow

1. Install Required Libraries
2. Load Pretrained TinyLlama Model
3. Analyze Model Parameters
4. Apply LoRA Adapters
5. Create Medical QA Dataset
6. Tokenize Medical Data
7. Fine-Tune LoRA Layers
8. Save LoRA Adapter
9. Load Fine-Tuned Model
10. Generate Medical Answers

---

## Technologies Used

### Programming Language

* Python

### Deep Learning Frameworks

* PyTorch
* Transformers
* PEFT (Parameter Efficient Fine-Tuning)
* Accelerate
* BitsAndBytes

### Model

* TinyLlama-1.1B-Chat-v1.0

### Dataset Library

* Hugging Face Datasets

### Environment

* Google Colab

---

## Dataset Format

Example dataset structure:

```python
data = {
    "question": [
        "What is diabetes?",
        "How to treat a common cold?",
        "What are the symptoms of hypertension?"
    ],
    "answer": [
        "Diabetes is a chronic disease where the body cannot regulate blood sugar properly.",
        "Rest, hydration, and over-the-counter medications can help manage symptoms.",
        "Symptoms include headaches, shortness of breath, and nosebleeds."
    ]
}
```

---

## Installation

Install the required dependencies:

```bash
pip install transformers datasets peft accelerate bitsandbytes sentencepiece
```

---

## Model Training

The pretrained TinyLlama model is adapted using LoRA.

LoRA Configuration:

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

### Advantages of LoRA

* Trains only a small subset of parameters
* Reduces GPU memory usage
* Faster training
* Smaller model checkpoints
* Cost-effective fine-tuning

---

## Running the Project

Load the fine-tuned model and ask medical questions:

```python
question = input("Enter your medical question: ")
response = medical_qa(question)
print(response)
```

Example:

Input:

```text
What is diabetes?
```

Output:

```text
Diabetes is a chronic disease in which the body cannot properly regulate blood sugar levels due to insufficient insulin production or ineffective insulin usage.
```

---

## Evaluation

### Before LoRA Fine-Tuning

* Generic responses
* Limited medical knowledge adaptation
* Less domain-specific understanding

### After LoRA Fine-Tuning

* Improved medical responses
* Better understanding of disease-related questions
* Domain-specific answer generation

---

## Results

The system successfully:

* Loaded a pretrained TinyLlama model
* Applied LoRA adapters
* Reduced trainable parameters to less than 1%
* Fine-tuned only adapter layers
* Generated medical-domain responses
* Saved lightweight LoRA adapters

---

## Future Enhancements

* Train on larger medical datasets
* Integrate medical knowledge bases
* Add symptom analysis support
* Support multilingual medical queries
* Deploy using Streamlit or Flask
* Integrate Retrieval-Augmented Generation (RAG)

---

## Applications

* Healthcare Chatbots
* Medical Information Assistants
* Patient Education Systems
* Clinical Support Tools
* Telemedicine Platforms
* Medical Research Assistance

---

## Conclusion

This project demonstrates how LoRA can efficiently adapt a pretrained Large Language Model for medical question answering. By training only a small number of parameters, the system achieves domain-specific performance while significantly reducing computational requirements compared to full fine-tuning.
