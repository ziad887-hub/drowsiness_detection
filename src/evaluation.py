def calculate_drowsiness_scores(predictions, timestamps=None):
    """
    Calculate drowsiness scores from model predictions.

    Args:
        predictions: List of prediction probabilities or binary classifications
        timestamps: Optional list of timestamps corresponding to predictions

    Returns:
        Dictionary containing drowsiness metrics
    """
    import numpy as np

    predictions = np.array(predictions)

    # Calculate basic metrics
    drowsy_ratio = np.mean(predictions > 0.5) if predictions.ndim > 0 else predictions > 0.5
    max_confidence = np.max(predictions) if predictions.size > 0 else 0.0
    avg_confidence = np.mean(predictions) if predictions.size > 0 else 0.0

    # Calculate temporal metrics if timestamps provided
    if timestamps is not None and len(timestamps) == len(predictions):
        timestamps = np.array(timestamps)
        # Time spent in drowsy state
        drowsy_periods = predictions > 0.5
        if np.any(drowsy_periods):
            # Find contiguous drowsy periods
            changes = np.diff(drowsy_periods.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            # Handle edge cases
            if drowsy_periods[0]:
                starts = np.insert(starts, 0, 0)
            if drowsy_periods[-1]:
                ends = np.append(ends, len(drowsy_periods))

            durations = timestamps[ends] - timestamps[starts]
            total_drowsy_time = np.sum(durations)
            longest_drowsy_period = np.max(durations) if len(durations) > 0 else 0
        else:
            total_drowsy_time = 0
            longest_drowsy_period = 0

        return {
            'drowsy_ratio': float(drowsy_ratio),
            'max_confidence': float(max_confidence),
            'avg_confidence': float(avg_confidence),
            'total_drowsy_time': float(total_drowsy_time),
            'longest_drowsy_period': float(longest_drowsy_period),
            'prediction_count': len(predictions)
        }
    else:
        return {
            'drowsy_ratio': float(drowsy_ratio),
            'max_confidence': float(max_confidence),
            'avg_confidence': float(avg_confidence),
            'prediction_count': len(predictions)
        }


def generate_report(results, output_path=None):
    """
    Generate a human-readable report from drowsiness detection results.

    Args:
        results: Dictionary of results from calculate_drowsiness_scores
        output_path: Optional path to save the report

    Returns:
        String containing the formatted report
    """
    report_lines = [
        "=" * 50,
        "DROWSINESS DETECTION REPORT",
        "=" * 50,
        f"Drowsy Ratio: {results.get('drowsy_ratio', 0):.2%}",
        f"Average Confidence: {results.get('avg_confidence', 0):.3f}",
        f"Maximum Confidence: {results.get('max_confidence', 0):.3f}",
    ]

    if 'total_drowsy_time' in results:
        report_lines.extend([
            f"Total Drowsy Time: {results['total_drowsy_time']:.2f} seconds",
            f"Longest Drowsy Period: {results['longest_drowsy_period']:.2f} seconds"
        ])

    report_lines.extend([
        f"Total Predictions Analyzed: {results.get('prediction_count', 0)}",
        "=" * 50
    ])

    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report


if __name__ == "__main__":
    # Example usage
    import numpy as np
    predictions = np.random.rand(100)  # Random predictions for demo
    results = calculate_drowsiness_scores(predictions)
    print(generate_report(results))