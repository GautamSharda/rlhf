import torch
from bert_score import BERTScorer
import json

# Initialize BERTScorer
scorer = BERTScorer(lang="en", rescale_with_baseline=True)

# Read the test results file
with open("test_results.txt", "r") as file:
    content = file.read()

# Split into before and after sections
sections = content.split("Responses after DPO feedback loop:")
before_section = sections[0].split("Responses before DPO feedback loop:\n")[1]
after_section = sections[1]

# Extract response pairs
def extract_responses(section):
    responses = []
    lines = section.strip().split('\n')
    current_response = ""
    for line in lines:
        if line.startswith("Prompt:"):
            if current_response:
                responses.append(current_response.strip())
            current_response = ""
        elif line.startswith("Response:"):
            current_response = line.replace("Response:", "").strip()
        elif current_response and line.strip():
            current_response += " " + line.strip()
    if current_response:
        responses.append(current_response.strip())
    return responses

before_responses = extract_responses(before_section)
after_responses = extract_responses(after_section)

# Calculate BERTScores
P, R, F1 = scorer.score(after_responses, before_responses)

# Print results
print("\nBERTScore Results:")
print("==================")
for i in range(len(before_responses)):
    print(f"\nPrompt {i+1}:")
    print(f"Precision: {P[i].item():.3f}")
    print(f"Recall: {R[i].item():.3f}")
    print(f"F1: {F1[i].item():.3f}")

# Calculate average scores
avg_P = P.mean().item()
avg_R = R.mean().item()
avg_F1 = F1.mean().item()

print("\nAverage Scores:")
print("===============")
print(f"Average Precision: {avg_P:.3f}")
print(f"Average Recall: {avg_R:.3f}")
print(f"Average F1: {avg_F1:.3f}")
