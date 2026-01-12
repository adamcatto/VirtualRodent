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

    # Get pagination params - no hard cap, user controls how much to load
    start = int(request.args.get('start', 0))
    limit = int(request.args.get('limit', 200))
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


# ============================================================================
# Raw Pose Data from HDF5 Files
# ============================================================================

@app.route("/api/sessions")
def list_sessions():
    """List all available sessions from HDF5 files."""
    import h5py
    
    sessions = []
    
    for brain_region in ["DLS", "motor_cortex"]:
        region_dir = DATA_DIR / brain_region
        if not region_dir.exists():
            continue
            
        for animal_dir in region_dir.iterdir():
            if not animal_dir.is_dir() or animal_dir.name.startswith('.'):
                continue
                
            animal = animal_dir.name
            
            for h5_file in animal_dir.glob("*.h5"):
                if h5_file.name.startswith('.'):
                    continue
                    
                session_id = h5_file.stem
                
                # Get basic info from file
                try:
                    with h5py.File(h5_file, 'r') as f:
                        num_frames = f['pose/keypoints'].shape[0]
                        sampling_rate = 50  # Hz
                        duration_seconds = num_frames / sampling_rate
                        
                        sessions.append({
                            "session_id": session_id,
                            "animal": animal,
                            "brain_region": brain_region,
                            "num_frames": num_frames,
                            "duration_seconds": duration_seconds,
                            "duration_minutes": duration_seconds / 60,
                            "sampling_rate_hz": sampling_rate,
                            "file_path": str(h5_file.relative_to(DATA_DIR)),
                        })
                except Exception as e:
                    print(f"Error reading {h5_file}: {e}")
    
    # Sort by session_id
    sessions.sort(key=lambda x: (x['brain_region'], x['animal'], x['session_id']))
    
    return jsonify(sessions)


@app.route("/api/session/<brain_region>/<animal>/<session_id>/info")
def get_raw_session_info(brain_region, animal, session_id):
    """Get info about a raw session from HDF5 file."""
    import h5py
    
    h5_file = DATA_DIR / brain_region / animal / f"{session_id}.h5"
    
    if not h5_file.exists():
        return jsonify({"error": "Session not found", "path_checked": str(h5_file)}), 404
    
    try:
        with h5py.File(h5_file, 'r') as f:
            keypoints_shape = f['pose/keypoints'].shape
            num_frames = keypoints_shape[0]
            num_coords = keypoints_shape[1]  # 3 (xyz)
            num_keypoints = keypoints_shape[2]  # 23
            
            sampling_rate = 50  # Hz
            duration_seconds = num_frames / sampling_rate
            
            return jsonify({
                "session_id": session_id,
                "animal": animal,
                "brain_region": brain_region,
                "num_frames": num_frames,
                "num_keypoints": num_keypoints,
                "num_coords": num_coords,
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_seconds / 60,
                "sampling_rate_hz": sampling_rate,
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<brain_region>/<animal>/<session_id>/pose")
def get_raw_pose(brain_region, animal, session_id):
    """Get raw pose keypoints from HDF5 file."""
    import h5py
    from flask import Response
    import json as json_module
    
    h5_file = DATA_DIR / brain_region / animal / f"{session_id}.h5"
    
    if not h5_file.exists():
        return jsonify({"error": "Session not found", "path_checked": str(h5_file)}), 404
    
    # Get pagination params
    start = int(request.args.get('start', 0))
    limit = int(request.args.get('limit', 1000))
    subsample = int(request.args.get('subsample', 1))  # Optional subsampling
    
    try:
        with h5py.File(h5_file, 'r') as f:
            keypoints = f['pose/keypoints']
            total_frames = keypoints.shape[0]
            
            end = min(start + limit, total_frames)
            
            # Load data with optional subsampling
            if subsample > 1:
                pose_data = keypoints[start:end:subsample, :, :]
                effective_rate = 50 / subsample
            else:
                pose_data = keypoints[start:end, :, :]
                effective_rate = 50
            
            # Convert from (T, 3, 23) to (T, 69) - flatten keypoints
            # Format: [x0, x1, ..., x22, y0, y1, ..., y22, z0, z1, ..., z22]
            T = pose_data.shape[0]
            pose_flat = np.zeros((T, 69), dtype=np.float32)
            pose_flat[:, :23] = pose_data[:, 0, :]  # x coords
            pose_flat[:, 23:46] = pose_data[:, 1, :]  # y coords
            pose_flat[:, 46:69] = pose_data[:, 2, :]  # z coords
            
            response = {
                "pose": pose_flat.tolist(),
                "num_frames_loaded": T,
                "total_frames": total_frames,
                "start": start,
                "end": end,
                "subsample": subsample,
                "effective_sampling_rate_hz": effective_rate,
                "original_sampling_rate_hz": 50,
                "duration_loaded_seconds": T / effective_rate,
                "total_duration_seconds": total_frames / 50,
            }
            
            json_str = json_module.dumps(response)
            size_mb = len(json_str) / 1024 / 1024
            print(f"Serving raw pose: {T} frames, {size_mb:.2f} MB")
            
            return Response(
                json_str,
                mimetype='application/json',
                headers={'Content-Length': str(len(json_str))}
            )
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting VirtualRodent API server...")
    print(f"Results directory: {RESULTS_DIR.absolute()}")
    print(f"Visualizations directory: {VIZ_DIR.absolute()}")
    print("\nAPI will be available at: http://localhost:5001")
    print("API documentation at: http://localhost:5001/")

    app.run(debug=True, host="0.0.0.0", port=5001)
