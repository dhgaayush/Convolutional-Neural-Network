#!/usr/bin/env python3

import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

PREPARE_SCRIPT = ROOT / "scripts" / "prepare_mnist.py"
CNN_BINARY = ROOT / "cnn"
RESULTS_DIR = ROOT / "experiments"
RESULTS_FILE = RESULTS_DIR / "results.csv"


# ============================================================
# HELPERS
# ============================================================

def run_command(command, cwd=ROOT):
    """Run a command and stop if it fails."""

    print("\n$", " ".join(map(str, command)))

    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    return result.stdout


def ask_int(prompt, default):
    """Ask for a positive integer with a default value."""

    while True:
        value = input(f"{prompt} [{default}]: ").strip()

        if value == "":
            return default

        try:
            value = int(value)

            if value <= 0:
                raise ValueError

            return value

        except ValueError:
            print("Please enter a positive integer.")


# ============================================================
# BUILD
# ============================================================

def build_cnn():
    print("\n" + "=" * 50)
    print("Building CNN")
    print("=" * 50)

    command = [
        "clang++",
        "-std=c++17",
        "-O2",
        "-Iincludes",
        *sorted(str(p) for p in (ROOT / "srcs").glob("*.cpp")),
        "-o",
        str(CNN_BINARY),
    ]

    pkg_config = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "opencv5"],
        text=True,
        capture_output=True
    )

    if pkg_config.returncode != 0:
        print(pkg_config.stderr, file=sys.stderr)
        sys.exit("Could not find OpenCV through pkg-config.")

    command.extend(pkg_config.stdout.split())

    run_command(command)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(train_limit, test_limit):
    print("\n" + "=" * 50)
    print("Preparing MNIST")
    print("=" * 50)

    command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--train-limit",
        str(train_limit),
        "--test-limit",
        str(test_limit),
    ]

    run_command(command)


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def run_experiment(train_limit, test_limit, experiment_name):
    model_path = f"{experiment_name}.bin"

    prepare_data(train_limit, test_limit)
    build_cnn()

    print("\n" + "=" * 50)
    print("Training")
    print("=" * 50)

    run_command([
        str(CNN_BINARY),
        "train",
        model_path
    ])

    print("\n" + "=" * 50)
    print("Evaluating")
    print("=" * 50)

    evaluation_output = run_command([
        str(CNN_BINARY),
        "eval",
        model_path
    ])

    match = re.search(
        r"Final Test Accuracy:\s*([0-9.]+)%",
        evaluation_output
    )

    if not match:
        print("\nWarning: Could not automatically extract test accuracy.")
        accuracy = ""
    else:
        accuracy = float(match.group(1))

    return model_path, accuracy


# ============================================================
# RESULTS LOGGING
# ============================================================

def log_result(
    experiment_name,
    train_limit,
    test_limit,
    model_path,
    accuracy
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = RESULTS_FILE.exists()

    with RESULTS_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "experiment",
                "train_samples",
                "test_samples",
                "test_accuracy",
                "model"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            experiment_name,
            train_limit,
            test_limit,
            accuracy,
            model_path
        ])

    print("\nResult recorded in:")
    print(RESULTS_FILE)


# ============================================================
# MAIN MENU
# ============================================================

def main():

    print("=" * 50)
    print("CNN Experiment Runner")
    print("=" * 50)

    train_limit = ask_int(
        "How many MNIST training images?",
        5000
    )

    test_limit = ask_int(
        "How many MNIST test images?",
        1000
    )

    default_name = f"model_{train_limit}"

    experiment_name = input(
        f"Experiment name [{default_name}]: "
    ).strip()

    if not experiment_name:
        experiment_name = default_name

    print("\nExperiment configuration:")

    print(f"  Training samples: {train_limit}")
    print(f"  Test samples:     {test_limit}")
    print(f"  Experiment:       {experiment_name}")

    confirm = input("\nRun experiment? [Y/n]: ").strip().lower()

    if confirm not in ("", "y", "yes"):
        print("Cancelled.")
        return

    model_path, accuracy = run_experiment(
        train_limit,
        test_limit,
        experiment_name
    )

    log_result(
        experiment_name,
        train_limit,
        test_limit,
        model_path,
        accuracy
    )

    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETE")
    print("=" * 50)

    print(f"Model:         {model_path}")
    print(f"Test accuracy: {accuracy}%")


if __name__ == "__main__":
    main()
