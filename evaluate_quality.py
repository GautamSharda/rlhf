import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)

def prepare_reward_model():
    """Prepare the OpenAssistant reward model for evaluation."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "OpenAssistant/reward-model-deberta-v3-large-v2",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
    )
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")
    return model, tokenizer

def evaluate_response_quality(response, prompt, reward_model, reward_tokenizer):
    """Evaluate a single response using the reward model."""
    full_text = f"Human: {prompt}\n\nAssistant: {response}"
    inputs = reward_tokenizer(full_text, return_tensors="pt", truncation=True).to("cuda")
    with torch.no_grad():
        score = reward_model(**inputs).logits[0].item()
    return score

def extract_prompt_response_pairs(section):
    """Extract prompt-response pairs from a section of the results file."""
    pairs = []
    lines = section.strip().split('\n')
    current_prompt = ""
    current_response = ""
    
    for line in lines:
        if line.startswith("Prompt:"):
            if current_prompt and current_response:
                pairs.append((current_prompt, current_response))
            current_prompt = line.replace("Prompt:", "").strip()
            current_response = ""
        elif line.startswith("Response:"):
            current_response = line.replace("Response:", "").strip()
        elif current_response and line.strip():
            current_response += " " + line.strip()
    
    if current_prompt and current_response:
        pairs.append((current_prompt, current_response))
    
    return pairs

def main():
    # Read test results file
    with open("test_results.txt", "r") as file:
        content = file.read()

    # Split into before and after sections
    sections = content.split("Responses after DPO feedback loop:")
    before_pairs = extract_prompt_response_pairs(sections[0].split("Responses before DPO feedback loop:\n")[1])
    after_pairs = extract_prompt_response_pairs(sections[1])

    # Initialize reward model
    reward_model, reward_tokenizer = prepare_reward_model()

    print("\nResponse Quality Evaluation:")
    print("===========================")

    total_pre_score = 0
    total_post_score = 0

    # Evaluate each pair
    for i in range(len(before_pairs)):
        prompt = before_pairs[i][0]
        pre_response = before_pairs[i][1]
        post_response = after_pairs[i][1]

        pre_score = evaluate_response_quality(pre_response, prompt, reward_model, reward_tokenizer)
        post_score = evaluate_response_quality(post_response, prompt, reward_model, reward_tokenizer)

        total_pre_score += pre_score
        total_post_score += post_score

        print(f"\nPrompt {i+1}: {prompt}")
        print("\nPre-DPO Response:")
        print(pre_response)
        print(f"Score: {pre_score:.3f}")
        
        print("\nPost-DPO Response:")
        print(post_response)
        print(f"Score: {post_score:.3f}")
        
        print(f"\nImprovement: {post_score - pre_score:.3f}")
        print("="*50)

    # Calculate averages
    avg_pre = total_pre_score / len(before_pairs)
    avg_post = total_post_score / len(after_pairs)

    print("\nOverall Results:")
    print("================")
    print(f"Average Pre-DPO Score: {avg_pre:.3f}")
    print(f"Average Post-DPO Score: {avg_post:.3f}")
    print(f"Average Improvement: {avg_post - avg_pre:.3f}")

if __name__ == "__main__":
    main()
