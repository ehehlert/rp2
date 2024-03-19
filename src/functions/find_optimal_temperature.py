from src.models.textract_pages.model__tune import evaluate_model_with_temp_scaling_and_ood_detection
from src.functions.calculate_ece import calculate_ece

def find_optimal_temperature(validation_set, model, temperatures):
    best_temperature = None
    best_criterion_score = float('inf')  # For ECE, lower is better. Adjust accordingly for other metrics.
    criterion_scores = []

    for T in temperatures:
        evaluated_set, y_prob = evaluate_model_with_temp_scaling_and_ood_detection(validation_set, model, temperature=T)
        # Assuming your evaluated_set DataFrame includes a column 'true_labels' for the true labels
        y_true = evaluated_set['true_labels'].to_numpy()  # You need to adjust this based on your actual DataFrame structure
        
        current_criterion_score = calculate_ece(y_true, y_prob)
        criterion_scores.append((T, current_criterion_score))
        
        if current_criterion_score < best_criterion_score:
            best_temperature = T
            best_criterion_score = current_criterion_score

    print(f"Best temperature found: {best_temperature}")
    print(f"Best ECE score: {best_criterion_score}")
    
    return best_temperature, criterion_scores
