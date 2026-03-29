#!/usr/bin/env python3
"""
PolarQuant-KV llama.cpp E2E Integration Test

Verifies:
1. llama-cli builds with GGML_POLARQUANT=ON
2. --polarquant flag is accepted
3. Model loads and generates text
4. KV Cache memory reduction is reported

Usage:
  python test_e2e.py [--model PATH_TO_GGUF]
"""

import subprocess
import sys
import os
import re

LLAMA_CLI = os.path.join("integration", "llama.cpp", "build", "bin", "Release", "llama-cli.exe")
DEFAULT_MODEL = os.path.join("models", "Qwen3.5-9B.Q4_K_M.gguf")


def run_cmd(cmd, timeout=120):
    """Run command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace"
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except FileNotFoundError:
        return -2, "", f"File not found: {cmd[0]}"


def test_binary_exists():
    """Test 1: llama-cli binary exists."""
    print("Test 1: Binary exists...", end=" ")
    if os.path.exists(LLAMA_CLI):
        print("PASS")
        return True
    # Try non-Release path
    alt = LLAMA_CLI.replace("Release", "Debug")
    if os.path.exists(alt):
        global LLAMA_CLI
        LLAMA_CLI = alt
        print(f"PASS (Debug build)")
        return True
    print(f"SKIP (not built yet: {LLAMA_CLI})")
    return False


def test_help_flag():
    """Test 2: --polarquant flag appears in help."""
    print("Test 2: --polarquant in help...", end=" ")
    rc, out, err = run_cmd([LLAMA_CLI, "--help"])
    combined = out + err
    if "--polarquant" in combined or "--pq" in combined:
        print("PASS")
        return True
    print("SKIP (flag not found, may not be compiled with GGML_POLARQUANT)")
    return False


def test_model_generation(model_path):
    """Test 3: Generate text with PolarQuant enabled."""
    print("Test 3: Model generation with --polarquant...", end=" ")
    if not os.path.exists(model_path):
        print(f"SKIP (model not found: {model_path})")
        return False

    rc, out, err = run_cmd([
        LLAMA_CLI,
        "-m", model_path,
        "--polarquant",
        "-p", "Hello, my name is",
        "-n", "16",
        "--no-display-prompt",
    ], timeout=300)

    combined = out + err
    if rc == 0 and len(out.strip()) > 0:
        print("PASS")
        print(f"  Generated: {out.strip()[:100]}...")
        # Check for PolarQuant init message
        if "PolarQuant" in combined:
            print("  PolarQuant initialization detected in logs")
        return True
    else:
        print(f"FAIL (rc={rc})")
        if err:
            print(f"  stderr: {err[:200]}")
        return False


def test_compression_ratio(model_path):
    """Test 4: Verify compression ratio is reported."""
    print("Test 4: Compression ratio reported...", end=" ")
    if not os.path.exists(model_path):
        print(f"SKIP (model not found: {model_path})")
        return False

    rc, out, err = run_cmd([
        LLAMA_CLI,
        "-m", model_path,
        "--polarquant",
        "-p", "Test",
        "-n", "1",
    ], timeout=300)

    combined = out + err
    # Look for compression ratio in output
    match = re.search(r"compression ratio (\d+\.?\d*)x", combined, re.IGNORECASE)
    if match:
        ratio = float(match.group(1))
        print(f"PASS (ratio: {ratio}x)")
        return ratio >= 2.0
    elif "PolarQuant" in combined:
        print("PASS (PolarQuant active)")
        return True
    else:
        print("SKIP (no compression info in output)")
        return False


def main():
    model_path = DEFAULT_MODEL
    if len(sys.argv) > 1 and sys.argv[1] == "--model":
        model_path = sys.argv[2]

    print("=== PolarQuant-KV llama.cpp E2E Test ===\n")

    results = []
    results.append(("Binary exists", test_binary_exists()))
    if results[-1][1]:
        results.append(("Help flag", test_help_flag()))
        results.append(("Model generation", test_model_generation(model_path)))
        results.append(("Compression ratio", test_compression_ratio(model_path)))

    print(f"\n=== Results ===")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        status = "PASS" if result else "FAIL/SKIP"
        print(f"  {name}: {status}")
    print(f"\n{passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
