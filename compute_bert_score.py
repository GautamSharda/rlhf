from evaluate import load
import numpy as np

def compute_bert_scores(before_responses, after_responses):
    bertscore = load("bertscore")
    
    # Calculate BERTScore between before and after responses
    results = bertscore.compute(
        predictions=after_responses,
        references=before_responses,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli"
    )
    
    # Calculate average scores
    avg_precision = np.mean(results['precision'])
    avg_recall = np.mean(results['recall'])
    avg_f1 = np.mean(results['f1'])
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }

if __name__ == "__main__":
    # Read the test results file
    with open("test_results.txt", "r") as f:
        content = f.read()
    
    # Split into before and after sections
    sections = content.split("Responses after DPO feedback loop:")
    before_section = sections[0].split("Responses before DPO feedback loop:\n")[1]
    after_section = sections[1]
    
    # Extract responses
    before_responses = []
    after_responses = []
    
    for section in before_section.split("Prompt:")[1:]:
        response = section.split("Response:")[1].strip().split("\n\n")[0]
        before_responses.append(response)
    
    for section in after_section.split("Prompt:")[1:]:
        response = section.split("Response:")[1].strip().split("\n\n")[0]
        after_responses.append(response)
    
    # Compute scores
    scores = compute_bert_scores(before_responses, after_responses)
    
    print("\nBERTScore Results:")
    print(f"Precision: {scores['precision']:.4f}")
    print(f"Recall: {scores['recall']:.4f}")
    print(f"F1: {scores['f1']:.4f}")