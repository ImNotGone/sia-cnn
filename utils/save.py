import json

results_dir = "results"

def save_errors_per_epoch(errors_per_epoch):
    with open(f"{results_dir}/errors_per_epoch.json", "w") as file:
        json.dump(errors_per_epoch, file, indent=4)

def save_errors_per_architecture(errors_per_architecture):
    with open(f"{results_dir}/errors_per_architecture.json", "w") as file:
        json.dump(errors_per_architecture, file, indent=4)


def save_predictions(predictions):
    predictions = [
        {
            "predicted": predicted,
            "actual": actual,
            "output": float(output),
            "label": int(label),
        }
        for predicted, actual, output, label in predictions
    ]
    with open(f"{results_dir}/predictions.json", "w") as file:
        json.dump(predictions, file, indent=4)
