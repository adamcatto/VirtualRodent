"""
Flask backend for VirtualRodent visualization website.

Provides REST API endpoints for accessing predictions, metrics, and neural data.
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from pathlib import Path
import json
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Get project root (src/website -> src -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Paths (absolute from project root)
RESULTS_DIR = PROJECT_ROOT / "outs/results"
VIZ_DIR = PROJECT_ROOT / "outs/visualizations"
DATA_DIR = PROJECT_ROOT / "data/Virtual_Rodent"


@app.route("/")
def index():
    """API root."""
    return jsonify({
        "message": "VirtualRodent API",
        "version": "1.0",
        "endpoints": [
            "/api/experiments",
            "/api/experiment/<name>/metrics",
            "/api/experiment/<name>/predictions",
            "/api/experiment/<name>/sessions",
            "/api/visualizations",
        ]
    })


def find_experiments_recursive(base_dir: Path, prefix: str = "") -> list:
    """
    Recursively find all experiments, including per-session subdirectories.

    Returns list of experiment dicts with hierarchical structure.
    """
    experiments = []

    if not base_dir.exists():
        return experiments

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        exp_name = f"{prefix}/{item.name}" if prefix else item.name
        results_file = item / "training_results.json"
        summary_file = item / "summary.json"

        # Check if this is a per-session parent directory (has summary.json)
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)

            # This is a per-session experiment with multiple sessions
            session_results = summary.get("session_results", [])
            experiments.append({
                "name": item.name,
                "strategy": "per_session",
                "model_type": summary.get("config", {}).get("model", {}).get("rnn_type", "lstm"),
                "is_multi_session": True,
                "total_sessions": summary.get("total_sessions", len(session_results)),
                "successful_sessions": summary.get("successful", len(session_results)),
                "sessions": [
                    {
                        "session_id": sr["session_id"],
                        "best_val_loss": sr.get("best_val_loss"),
                        "test_metrics": sr.get("test_metrics", {}),
                        "session_metadata": sr.get("session_metadata", {}),
                    }
                    for sr in session_results
                ],
                "avg_val_loss": np.mean([sr["best_val_loss"] for sr in session_results if sr.get("best_val_loss")]) if session_results else None,
            })

        # Check if this is a single experiment (has training_results.json)
        elif results_file.exists():
            with open(results_file) as f:
                results = json.load(f)

            experiments.append({
                "name": exp_name,
                "strategy": results.get("strategy"),
                "model_type": results.get("model_type"),
                "input_mode": results.get("input_mode"),
                "best_val_loss": results.get("best_val_loss"),
                "session_id": results.get("session_id"),  # For single-session experiments
                "session_metadata": results.get("session_metadata"),
                "is_multi_session": False,
            })

        # Check if this is an in-progress multi-session experiment (has subdirs with training_results.json)
        else:
            # Look for session subdirectories with training_results.json
            session_results = []
            for subdir in item.iterdir():
                if subdir.is_dir():
                    sub_results_file = subdir / "training_results.json"
                    if sub_results_file.exists():
                        with open(sub_results_file) as f:
                            sr = json.load(f)
                        session_results.append({
                            "session_id": sr.get("session_id", subdir.name),
                            "best_val_loss": sr.get("best_val_loss"),
                            "test_metrics": sr.get("test_metrics", {}),
                            "session_metadata": sr.get("session_metadata", {}),
                        })

            if session_results:
                # Get model type from first session's config
                first_session_file = item / session_results[0]["session_id"] / "training_results.json"
                model_type = "lstm"
                if first_session_file.exists():
                    with open(first_session_file) as f:
                        first_config = json.load(f)
                    model_type = first_config.get("model_type", "lstm")

                experiments.append({
                    "name": item.name,
                    "strategy": "per_session",
                    "model_type": model_type,
                    "is_multi_session": True,
                    "total_sessions": len(session_results),
                    "successful_sessions": len(session_results),
                    "sessions": session_results,
                    "avg_val_loss": np.mean([sr["best_val_loss"] for sr in session_results if sr.get("best_val_loss")]) if session_results else None,
                })

    return experiments


@app.route("/api/experiments")
def get_experiments():
    """List all experiments, including per-session hierarchical structure."""
    experiments = find_experiments_recursive(RESULTS_DIR)
    return jsonify(experiments)


@app.route("/api/experiment/<name>/metrics")
def get_experiment_metrics(name):
    """Get metrics for a specific experiment."""
    # Check for per-session summary first
    summary_file = RESULTS_DIR / name / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        return jsonify({
            "is_multi_session": True,
            "summary": summary,
        })

    # Check for single experiment results
    results_file = RESULTS_DIR / name / "training_results.json"
    if not results_file.exists():
        return jsonify({"error": "Experiment not found"}), 404

    with open(results_file) as f:
        results = json.load(f)

    # Also load overall metrics if available
    overall_metrics_file = RESULTS_DIR / name / "overall_metrics.json"
    if overall_metrics_file.exists():
        with open(overall_metrics_file) as f:
            results["overall_metrics"] = json.load(f)

    results["is_multi_session"] = False
    return jsonify(results)


@app.route("/api/experiment/<name>/session/<session_id>/metrics")
def get_session_metrics(name, session_id):
    """Get metrics for a specific session within a per-session experiment."""
    results_file = RESULTS_DIR / name / session_id / "training_results.json"

    if not results_file.exists():
        return jsonify({"error": "Session not found"}), 404

    with open(results_file) as f:
        results = json.load(f)

    return jsonify(results)


@app.route("/api/experiment/<name>/predictions")
def get_predictions(name):
    """Get predictions for visualization."""
    pred_file = RESULTS_DIR / name / "predictions" / "test_predictions.npz"

    if not pred_file.exists():
        return jsonify({"error": "Predictions not found"}), 404

    # Load predictions
    data = np.load(pred_file)

    # Return first N samples for visualization (to avoid huge payload)
    num_samples = min(100, len(data['predictions']))

    return jsonify({
        "predictions": data['predictions'][:num_samples].tolist(),
        "ground_truth": data['ground_truth'][:num_samples].tolist(),
        "num_total_samples": len(data['predictions']),
    })


@app.route("/api/experiment/<name>/session/<session_id>/info")
def get_session_info(name, session_id):
    """Get session info without loading full data - for pre-load UI."""
    pred_file = RESULTS_DIR / name / session_id / "predictions" / "test_predictions.npz"

    if not pred_file.exists():
        return jsonify({"error": "Predictions not found", "path_checked": str(pred_file)}), 404

    # Load just metadata (numpy loads lazily)
    data = np.load(pred_file)
    
    num_samples = len(data['predictions'])
    pose_horizon = data['predictions'].shape[1]
    pose_dim = data['predictions'].shape[2]
    
    neural_info = {}
    if 'neural' in data and len(data['neural']) > 0:
        neural_info = {
            "neural_history": data['neural'].shape[1],
            "num_neurons": data['neural'].shape[2],
        }
    
    # Calculate total time in seconds (50Hz sampling)
    total_time_seconds = num_samples / 50.0
    
    return jsonify({
        "num_samples": num_samples,
        "pose_horizon": pose_horizon,
        "pose_dim": pose_dim,
        "total_time_seconds": total_time_seconds,
        "sampling_rate_hz": 50,
        **neural_info,
    })


@app.route("/api/experiment/<name>/session/<session_id>/predictions")
def get_session_predictions(name, session_id):
    """Get predictions for a specific session with neural data."""
    from flask import Response
    import json as json_module
    
    pred_file = RESULTS_DIR / name / session_id / "predictions" / "test_predictions.npz"

    if not pred_file.exists():
        return jsonify({"error": "Predictions not found", "path_checked": str(pred_file)}), 404

    # Load predictions
    data = np.load(pred_file)

    # Get pagination params
    start = int(request.args.get('start', 0))
    limit = int(request.args.get('limit', 200))
    
    # IMPORTANT: Cap samples to keep JSON payload under ~50MB
    # Full data: 1000 samples × 250 timesteps × 69 coords × 2 (pred+gt) = ~500MB JSON
    # With caps: 200 samples × 50 timesteps × 69 coords × 2 = ~20MB JSON
    MAX_SAMPLES = 200
    limit = min(limit, MAX_SAMPLES)
    end = min(start + limit, len(data['predictions']))

    # Subsample pose horizon (250 -> 50 timesteps)
    pose_subsample = int(request.args.get('pose_subsample', 5))
    
    # Subsample neural time (750 -> 75 timesteps)  
    neural_time_subsample = int(request.args.get('neural_subsample', 10))
    
    # Subsample neurons (131 -> 50 neurons max)
    max_neurons = int(request.args.get('max_neurons', 50))

    actual_samples = end - start
    print(f"Serving predictions: samples {start}-{end} ({actual_samples} samples)")
    print(f"  Pose subsample: {pose_subsample}x, Neural time subsample: {neural_time_subsample}x, Max neurons: {max_neurons}")

    # Subsample pose horizon to reduce payload
    predictions_sub = data['predictions'][start:end, ::pose_subsample, :]
    ground_truth_sub = data['ground_truth'][start:end, ::pose_subsample, :]
    
    response = {
        "predictions": predictions_sub.tolist(),
        "ground_truth": ground_truth_sub.tolist(),
        "num_total_samples": len(data['predictions']),
        "start": start,
        "end": end,
        "limit_applied": limit,
        "pose_subsample_factor": pose_subsample,
        "original_pose_horizon": data['predictions'].shape[1],
    }

    # Always include neural data (required)
    if 'neural' in data and len(data['neural']) > 0:
        # Subsample time and limit neurons
        neural_subset = data['neural'][start:end, ::neural_time_subsample, :max_neurons]
        response["neural"] = neural_subset.tolist()
        response["neural_subsampled"] = True
        response["neural_time_subsample_factor"] = neural_time_subsample
        response["neural_neurons_limited"] = min(max_neurons, data['neural'].shape[2])
        response["original_num_neurons"] = data['neural'].shape[2]

    # Serialize to JSON
    json_str = json_module.dumps(response)
    size_mb = len(json_str) / 1024 / 1024
    print(f"  Response size: {size_mb:.2f} MB")
    
    if size_mb > 50:
        print(f"  WARNING: Response size ({size_mb:.2f} MB) exceeds 50MB, may cause issues!")

    return Response(
        json_str,
        mimetype='application/json',
        headers={
            'Content-Length': str(len(json_str)),
        }
    )


@app.route("/api/visualizations")
def get_visualizations():
    """List available visualizations."""
    visualizations = {
        "plots": [],
        "videos": [],
        "neural": [],
    }

    if VIZ_DIR.exists():
        for subdir in ["plots", "videos", "neural"]:
            subdir_path = VIZ_DIR / subdir
            if subdir_path.exists():
                visualizations[subdir] = [
                    f.name for f in subdir_path.glob("*")
                    if f.is_file() and f.suffix in ['.png', '.jpg', '.pdf', '.mp4', '.gif']
                ]

    return jsonify(visualizations)


@app.route("/api/visualization/<category>/<filename>")
def get_visualization_file(category, filename):
    """Serve a visualization file."""
    file_path = VIZ_DIR / category / filename

    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path)


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    print("Starting VirtualRodent API server...")
    print(f"Results directory: {RESULTS_DIR.absolute()}")
    print(f"Visualizations directory: {VIZ_DIR.absolute()}")
    print("\nAPI will be available at: http://localhost:5001")
    print("API documentation at: http://localhost:5001/")

    app.run(debug=True, host="0.0.0.0", port=5001)
