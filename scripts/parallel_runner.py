#!/usr/bin/env python3
"""
Parallel Experiment Runner
==========================

Launch multiple research experiments in parallel on separate RunPod pods.

Usage:
    python parallel_runner.py --layer-sweep      # Test layers 10,15,18,20,22,24,27
    python parallel_runner.py --alpha-sweep      # Test α=-1,-2,-5,-8,-10 at layer 21
    python parallel_runner.py --both             # Run both sweeps (12 pods)
    python parallel_runner.py --status           # Check status of running experiments
    python parallel_runner.py --collect          # Collect all results
    python parallel_runner.py --terminate        # Terminate all pods

Environment variables required:
    RUNPOD_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, XAI_API_KEY
"""

import os
import sys
import json
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Install requests if needed
try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

SCRIPT_DIR = Path(__file__).parent
STATE_FILE = SCRIPT_DIR / ".parallel_state.json"
RESULTS_DIR = SCRIPT_DIR / "parallel_results"

# RunPod configuration
RUNPOD_API = "https://api.runpod.io/graphql"
GPU_CONFIG = {
    "name": "RTX A6000",
    "gpu_type_id": "NVIDIA RTX A6000",
    "cloud_type": "SECURE",
    "min_vcpu": 8,
    "min_memory": 32,
    "volume_gb": 100,
    "cost_per_hour": 0.79
}
DOCKER_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

# Experiment configurations
LAYER_SWEEP = [10, 15, 18, 20, 22, 24, 27]  # α=-3.0 fixed
ALPHA_SWEEP = [-1.0, -2.0, -5.0, -8.0, -10.0]  # layer=21 fixed

# SSH key location
SSH_KEY = SCRIPT_DIR / ".ssh" / "runpod_key"


def get_api_keys():
    """Get all required API keys."""
    return {
        "RUNPOD_API_KEY": os.environ.get("RUNPOD_API_KEY", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
        "XAI_API_KEY": os.environ.get("XAI_API_KEY", ""),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    }


def runpod_query(query: str, api_key: str) -> dict:
    """Execute RunPod GraphQL query."""
    response = requests.post(
        RUNPOD_API,
        json={"query": query},
        params={"api_key": api_key}
    )
    response.raise_for_status()
    return response.json().get("data", {})


def create_pod(experiment_id: str, layer: int, alpha: float, api_key: str) -> dict:
    """Create a RunPod pod for an experiment."""

    query = f"""
    mutation {{
        podFindAndDeployOnDemand(input: {{
            name: "exp-{experiment_id}"
            imageName: "{DOCKER_IMAGE}"
            gpuTypeId: "{GPU_CONFIG['gpu_type_id']}"
            cloudType: {GPU_CONFIG['cloud_type']}
            volumeInGb: {GPU_CONFIG['volume_gb']}
            containerDiskInGb: 20
            minVcpuCount: {GPU_CONFIG['min_vcpu']}
            minMemoryInGb: {GPU_CONFIG['min_memory']}
            startSsh: true
        }}) {{
            id
            name
            desiredStatus
            runtime {{
                ports {{
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                }}
            }}
        }}
    }}
    """

    result = runpod_query(query, api_key)
    pod = result.get("podFindAndDeployOnDemand", {})

    return {
        "pod_id": pod.get("id"),
        "name": pod.get("name"),
        "layer": layer,
        "alpha": alpha,
        "status": "creating",
        "created_at": datetime.now().isoformat()
    }


def get_pod_status(pod_id: str, api_key: str) -> dict:
    """Get pod status and SSH info."""
    query = f"""
    query {{
        pod(input: {{ podId: "{pod_id}" }}) {{
            id
            name
            desiredStatus
            runtime {{
                ports {{
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                }}
            }}
        }}
    }}
    """
    result = runpod_query(query, api_key)
    return result.get("pod", {})


def terminate_pod(pod_id: str, api_key: str):
    """Terminate a pod."""
    query = f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}'
    runpod_query(query, api_key)


def wait_for_pod_ready(pod_id: str, api_key: str, timeout: int = 300) -> dict:
    """Wait for pod to be ready with SSH access."""
    start = time.time()
    while time.time() - start < timeout:
        pod = get_pod_status(pod_id, api_key)
        runtime = pod.get("runtime", {})
        ports = runtime.get("ports", []) if runtime else []

        ssh_port = None
        ssh_ip = None
        for port in ports:
            if port.get("privatePort") == 22:
                ssh_port = port.get("publicPort")
                ssh_ip = port.get("ip")
                break

        if ssh_ip and ssh_port:
            return {"ip": ssh_ip, "port": ssh_port}

        time.sleep(10)

    raise TimeoutError(f"Pod {pod_id} not ready after {timeout}s")


def run_experiment_on_pod(exp: dict, api_keys: dict) -> dict:
    """Upload and run experiment on a pod."""
    pod_id = exp["pod_id"]
    layer = exp["layer"]
    alpha = exp["alpha"]

    try:
        # Wait for pod to be ready
        ssh_info = wait_for_pod_ready(pod_id, api_keys["RUNPOD_API_KEY"])
        exp["ssh_ip"] = ssh_info["ip"]
        exp["ssh_port"] = ssh_info["port"]
        exp["status"] = "ready"

        ssh_cmd = f"ssh -i {SSH_KEY} -p {ssh_info['port']} -o StrictHostKeyChecking=no root@{ssh_info['ip']}"

        # Upload pipeline.py
        scp_cmd = f"scp -i {SSH_KEY} -P {ssh_info['port']} -o StrictHostKeyChecking=no {SCRIPT_DIR}/pipeline.py root@{ssh_info['ip']}:/workspace/"
        subprocess.run(scp_cmd, shell=True, check=True, capture_output=True)

        # Install dependencies and set up environment
        setup_cmds = f"""
        pip install -q transformers accelerate bitsandbytes openai anthropic google-generativeai tqdm
        export HF_HOME=/workspace/.cache/huggingface
        export OPENAI_API_KEY='{api_keys['OPENAI_API_KEY']}'
        export ANTHROPIC_API_KEY='{api_keys['ANTHROPIC_API_KEY']}'
        export GOOGLE_API_KEY='{api_keys['GOOGLE_API_KEY']}'
        export XAI_API_KEY='{api_keys['XAI_API_KEY']}'
        huggingface-cli login --token '{api_keys['HF_TOKEN']}' 2>/dev/null || true
        """

        subprocess.run(f'{ssh_cmd} "{setup_cmds}"', shell=True, capture_output=True)

        # Run experiment with nohup
        exp_id = f"L{layer}_A{str(alpha).replace('-', 'n').replace('.', 'p')}"
        run_cmd = f"""
        cd /workspace
        export HF_HOME=/workspace/.cache/huggingface
        export OPENAI_API_KEY='{api_keys['OPENAI_API_KEY']}'
        export ANTHROPIC_API_KEY='{api_keys['ANTHROPIC_API_KEY']}'
        export GOOGLE_API_KEY='{api_keys['GOOGLE_API_KEY']}'
        export XAI_API_KEY='{api_keys['XAI_API_KEY']}'
        nohup python pipeline.py --layer {layer} --alpha {alpha} --output-dir /workspace/output --no-resume > /workspace/output/run.log 2>&1 &
        echo $!
        """

        result = subprocess.run(f'{ssh_cmd} "{run_cmd}"', shell=True, capture_output=True, text=True)
        exp["pid"] = result.stdout.strip()
        exp["status"] = "running"
        exp["experiment_id"] = exp_id

        print(f"  ✅ Started: Layer {layer}, α={alpha} on pod {pod_id}")

    except Exception as e:
        exp["status"] = "error"
        exp["error"] = str(e)
        print(f"  ❌ Failed: Layer {layer}, α={alpha}: {e}")

    return exp


def launch_experiments(experiments: list, api_keys: dict) -> list:
    """Launch all experiments in parallel."""

    print(f"\n🚀 Launching {len(experiments)} experiments...")
    print(f"   Estimated cost: ${len(experiments) * GPU_CONFIG['cost_per_hour']:.2f}/hr")
    print()

    # Create all pods first
    for i, exp in enumerate(experiments):
        exp_id = f"L{exp['layer']}_A{str(exp['alpha']).replace('-', 'n').replace('.', 'p')}"
        print(f"  Creating pod {i+1}/{len(experiments)}: Layer {exp['layer']}, α={exp['alpha']}...")

        try:
            pod_info = create_pod(exp_id, exp["layer"], exp["alpha"], api_keys["RUNPOD_API_KEY"])
            exp.update(pod_info)
            print(f"    → Pod ID: {pod_info['pod_id']}")
        except Exception as e:
            exp["status"] = "error"
            exp["error"] = str(e)
            print(f"    ❌ Error: {e}")

    # Save state
    save_state(experiments)

    # Wait a bit for pods to initialize
    print("\n⏳ Waiting 60s for pods to initialize...")
    time.sleep(60)

    # Run experiments in parallel
    print("\n📤 Uploading and starting experiments...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_experiment_on_pod, exp, api_keys): exp for exp in experiments if exp.get("pod_id")}
        for future in as_completed(futures):
            exp = futures[future]
            try:
                future.result()
            except Exception as e:
                exp["status"] = "error"
                exp["error"] = str(e)

    # Save final state
    save_state(experiments)

    return experiments


def check_experiment_status(exp: dict, api_keys: dict) -> dict:
    """Check if an experiment is still running."""
    if exp.get("status") not in ["running", "ready"]:
        return exp

    try:
        ssh_cmd = f"ssh -i {SSH_KEY} -p {exp['ssh_port']} -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@{exp['ssh_ip']}"

        # Check if process is still running
        result = subprocess.run(
            f'{ssh_cmd} "ps aux | grep -v grep | grep pipeline.py || echo DONE"',
            shell=True, capture_output=True, text=True, timeout=30
        )

        if "DONE" in result.stdout:
            # Check for results
            result = subprocess.run(
                f'{ssh_cmd} "cat /workspace/output/final_report.json 2>/dev/null || echo NO_RESULTS"',
                shell=True, capture_output=True, text=True, timeout=30
            )

            if "NO_RESULTS" not in result.stdout:
                exp["status"] = "completed"
                exp["results"] = json.loads(result.stdout)
            else:
                # Check log for errors
                result = subprocess.run(
                    f'{ssh_cmd} "tail -20 /workspace/output/run.log 2>/dev/null"',
                    shell=True, capture_output=True, text=True, timeout=30
                )
                exp["status"] = "failed"
                exp["log_tail"] = result.stdout
        else:
            # Still running - get progress
            result = subprocess.run(
                f'{ssh_cmd} "tail -5 /workspace/output/run.log 2>/dev/null"',
                shell=True, capture_output=True, text=True, timeout=30
            )
            exp["log_tail"] = result.stdout

    except Exception as e:
        exp["last_check_error"] = str(e)

    return exp


def save_state(experiments: list):
    """Save experiment state to file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(experiments, f, indent=2, default=str)


def load_state() -> list:
    """Load experiment state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return []


def collect_results(experiments: list, api_keys: dict):
    """Collect results from all completed experiments."""
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n📥 Collecting results...")

    for exp in experiments:
        if exp.get("status") != "completed":
            continue

        exp_id = exp.get("experiment_id", f"L{exp['layer']}_A{exp['alpha']}")
        result_dir = RESULTS_DIR / exp_id
        result_dir.mkdir(exist_ok=True)

        try:
            ssh_cmd = f"ssh -i {SSH_KEY} -p {exp['ssh_port']} -o StrictHostKeyChecking=no root@{exp['ssh_ip']}"

            # Download results via tar
            subprocess.run(
                f'{ssh_cmd} "cd /workspace/output && tar czf - ." | tar xzf - -C {result_dir}/',
                shell=True, check=True
            )
            print(f"  ✅ Collected: {exp_id}")

        except Exception as e:
            print(f"  ❌ Failed to collect {exp_id}: {e}")


def terminate_all(experiments: list, api_keys: dict):
    """Terminate all pods."""
    print("\n🛑 Terminating all pods...")

    for exp in experiments:
        pod_id = exp.get("pod_id")
        if pod_id:
            try:
                terminate_pod(pod_id, api_keys["RUNPOD_API_KEY"])
                print(f"  ✅ Terminated: {pod_id}")
                exp["status"] = "terminated"
            except Exception as e:
                print(f"  ❌ Failed to terminate {pod_id}: {e}")

    save_state(experiments)


def print_status(experiments: list):
    """Print status of all experiments."""
    print("\n" + "="*70)
    print("EXPERIMENT STATUS")
    print("="*70)

    for exp in experiments:
        layer = exp.get("layer", "?")
        alpha = exp.get("alpha", "?")
        status = exp.get("status", "unknown")
        pod_id = exp.get("pod_id", "N/A")[:12] if exp.get("pod_id") else "N/A"

        status_icon = {
            "completed": "✅",
            "running": "🔄",
            "ready": "⏳",
            "creating": "🔨",
            "error": "❌",
            "failed": "💥",
            "terminated": "🛑"
        }.get(status, "❓")

        print(f"  {status_icon} Layer {layer:2}, α={alpha:5} | {status:12} | Pod: {pod_id}")

        if status == "completed" and exp.get("results"):
            results = exp["results"]
            verdict = results.get("experiment_results", {}).get("final_verdict", "?")
            flip_rate = results.get("experiment_results", {}).get("control3", {}).get("flip_rate", 0)
            print(f"       → Verdict: {verdict}, Flip rate: {flip_rate*100:.1f}%")

    # Summary
    completed = sum(1 for e in experiments if e.get("status") == "completed")
    running = sum(1 for e in experiments if e.get("status") == "running")
    failed = sum(1 for e in experiments if e.get("status") in ["error", "failed"])

    print()
    print(f"  Total: {len(experiments)} | Completed: {completed} | Running: {running} | Failed: {failed}")
    print("="*70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parallel Experiment Runner")
    parser.add_argument("--layer-sweep", action="store_true", help="Run layer sweep")
    parser.add_argument("--alpha-sweep", action="store_true", help="Run alpha sweep")
    parser.add_argument("--both", action="store_true", help="Run both sweeps")
    parser.add_argument("--status", action="store_true", help="Check experiment status")
    parser.add_argument("--collect", action="store_true", help="Collect results")
    parser.add_argument("--terminate", action="store_true", help="Terminate all pods")
    args = parser.parse_args()

    api_keys = get_api_keys()

    if not api_keys["RUNPOD_API_KEY"]:
        print("❌ RUNPOD_API_KEY not set")
        sys.exit(1)

    if args.status:
        experiments = load_state()
        if not experiments:
            print("No experiments found. Run with --layer-sweep, --alpha-sweep, or --both")
            return

        for exp in experiments:
            check_experiment_status(exp, api_keys)
        save_state(experiments)
        print_status(experiments)
        return

    if args.collect:
        experiments = load_state()
        collect_results(experiments, api_keys)
        return

    if args.terminate:
        experiments = load_state()
        terminate_all(experiments, api_keys)
        return

    # Build experiment list
    experiments = []

    if args.layer_sweep or args.both:
        for layer in LAYER_SWEEP:
            experiments.append({"layer": layer, "alpha": -3.0})

    if args.alpha_sweep or args.both:
        for alpha in ALPHA_SWEEP:
            experiments.append({"layer": 21, "alpha": alpha})

    if not experiments:
        print("Specify --layer-sweep, --alpha-sweep, or --both")
        parser.print_help()
        return

    # Launch experiments
    experiments = launch_experiments(experiments, api_keys)
    print_status(experiments)

    print("\n💡 Monitor with: python parallel_runner.py --status")
    print("💡 Collect with: python parallel_runner.py --collect")
    print("💡 Terminate with: python parallel_runner.py --terminate")


if __name__ == "__main__":
    main()
