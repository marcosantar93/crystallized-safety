#!/usr/bin/env python3
"""
Launch Empathy Experiment on RunPod - 5 Pods in Parallel
Maximum speed: Each model on its own GPU pod
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "runpod_experiments"))

from runpod_graphql_orchestrator import create_pod, get_pod_status
import json
import time
from datetime import datetime

# Docker image with empathy experiment code baked in
DOCKER_IMAGE = "marcosantar93/crystallized-safety:empathy"

MODELS = {
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2-9b": "google/gemma-2-9b-it",
    "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Llama-7B"
}

def launch_model_pod(model_name: str, model_path: str) -> str:
    """Launch a pod for a single model"""
    print(f"🚀 Launching pod for: {model_name}")

    env_vars = {
        'EXPERIMENT_TYPE': 'empathy_geometry',
        'MODEL_NAME': model_name,
        'MODEL_PATH': model_path,
        'OUTPUT_DIR': '/workspace/results'
    }

    try:
        # Try multiple GPU types for availability
        gpu_types = ["NVIDIA RTX A5000", "NVIDIA RTX 4090", "NVIDIA RTX 3090"]

        for gpu_type in gpu_types:
            try:
                pod_id = create_pod(
                    name=f"empathy-{model_name}",
                    gpu_type=gpu_type,
                    env_vars=env_vars,
                    docker_image=DOCKER_IMAGE
                )
                if pod_id:
                    print(f"  ✅ Pod created: {pod_id} (GPU: {gpu_type})")
                    return pod_id
            except Exception as e:
                if "does not have the resources" in str(e):
                    continue
                else:
                    print(f"  ⚠️ Error with {gpu_type}: {e}")
                    continue

        print(f"  ❌ No GPU available for {model_name}")
        return None

    except Exception as e:
        print(f"  ❌ Failed to launch {model_name}: {e}")
        return None

def monitor_pods(pod_ids: dict) -> dict:
    """Monitor all pods until completion"""
    print("\n" + "="*80)
    print("MONITORING PODS")
    print("="*80)

    start_time = time.time()
    completed = {}

    while len(completed) < len(pod_ids):
        for model_name, pod_id in pod_ids.items():
            if model_name in completed:
                continue

            try:
                status = get_pod_status(pod_id)
                if not status:
                    continue

                desired = status.get('desiredStatus', 'UNKNOWN')
                runtime = status.get('runtime', {})
                uptime = runtime.get('uptimeInSeconds', 0)

                if desired == 'EXITED':
                    elapsed = time.time() - start_time
                    print(f"✅ {model_name}: COMPLETED ({elapsed:.1f}s)")
                    completed[model_name] = {
                        'status': 'completed',
                        'elapsed': elapsed,
                        'pod_id': pod_id
                    }
                elif uptime > 3600:  # 1 hour timeout
                    print(f"⏱️  {model_name}: TIMEOUT (>{uptime}s)")
                    completed[model_name] = {
                        'status': 'timeout',
                        'elapsed': uptime,
                        'pod_id': pod_id
                    }
                else:
                    # Still running
                    print(f"⏳ {model_name}: Running ({uptime}s)...", end='\r')

            except Exception as e:
                print(f"❌ {model_name}: Error monitoring - {e}")
                completed[model_name] = {
                    'status': 'error',
                    'error': str(e),
                    'pod_id': pod_id
                }

        if len(completed) < len(pod_ids):
            time.sleep(30)  # Check every 30 seconds

    return completed

def main():
    print("="*80)
    print("EMPATHY GEOMETRY - RUNPOD PARALLEL EXECUTION")
    print("="*80)
    print(f"Models: {len(MODELS)}")
    print(f"Strategy: 1 pod per model, all running in parallel")
    print()

    # Launch all pods
    print("Launching all pods in parallel...")
    pod_ids = {}

    for model_name, model_path in MODELS.items():
        pod_id = launch_model_pod(model_name, model_path)
        if pod_id:
            pod_ids[model_name] = pod_id
        time.sleep(2)  # Small delay between launches

    if not pod_ids:
        print("\n❌ No pods launched successfully!")
        return

    print(f"\n✅ Launched {len(pod_ids)}/{len(MODELS)} pods")
    print()

    # Save pod info
    pod_info_file = Path(__file__).parent / "results" / "empathy" / "pod_info.json"
    pod_info_file.parent.mkdir(exist_ok=True, parents=True)

    pod_info = {
        "timestamp": datetime.now().isoformat(),
        "pods": pod_ids,
        "models": MODELS
    }

    with open(pod_info_file, 'w') as f:
        json.dump(pod_info, f, indent=2)

    print(f"📊 Pod info saved to: {pod_info_file}")
    print()

    # Monitor pods
    results = monitor_pods(pod_ids)

    # Summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)

    successful = [m for m, r in results.items() if r['status'] == 'completed']
    failed = [m for m, r in results.items() if r['status'] != 'completed']

    print(f"✅ Successful: {len(successful)}/{len(pod_ids)}")
    print(f"❌ Failed: {len(failed)}/{len(pod_ids)}")
    print()

    if successful:
        print("Successful models:")
        for model in successful:
            elapsed = results[model]['elapsed']
            print(f"  - {model}: {elapsed:.1f}s")
        print()

    # Save results log
    results_log = Path(__file__).parent / "results" / "empathy" / f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_log, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "pods": pod_ids,
            "results": results,
            "summary": {
                "successful": len(successful),
                "failed": len(failed),
                "total": len(pod_ids)
            }
        }, f, indent=2)

    print(f"📊 Results log saved to: {results_log}")
    print()

    # Next steps
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. Download results from pods:")
    for model, pod_id in pod_ids.items():
        print(f"   ssh pod-{pod_id} 'cat /workspace/results/*.json'")
    print()
    print("2. Analyze results: python3 analyze_empathy_results.py")
    print("3. Generate PDF: python3 create_empathy_report.py")
    print()

if __name__ == "__main__":
    main()
