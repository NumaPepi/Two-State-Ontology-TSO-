#!/usr/bin/env python3
"""
TSO Validation Script
=====================
Tests Two State Ontology predictions on quantum hardware.

Usage:
    python tso_validation.py --backend aer_simulator
    python tso_validation.py --backend ibm_marrakesh --shots 4096

Author: John Pepin (incapp.org)
"""

import argparse
import json
import numpy as np
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Try importing IBM Runtime (optional for simulator-only use)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    HAS_IBM_RUNTIME = True
except ImportError:
    HAS_IBM_RUNTIME = False
    print("Note: qiskit-ibm-runtime not installed. Simulator only.")


# =============================================================================
# TSO CONSTANTS (Derived from percolation theory, NOT fitted)
# =============================================================================

P_C = 0.3116    # 3D site percolation threshold
N_C = 6         # Cubic lattice coordination number
E = np.e        # Euler's number


def compute_alpha(rho_c=0.5):
    """TSO coupling constant: α = 7/(1 + e×ρ_c)"""
    return 7.0 / (1.0 + E * rho_c)


def tso_entropy(p, s_max=1.0, kappa=1.35):
    """TSO prediction for entropy vs measurement rate."""
    x = kappa * N_C * (P_C - p) / P_C
    return s_max * (1.0 + np.tanh(x)) / 2.0


# =============================================================================
# QUANTUM CIRCUITS
# =============================================================================

def build_mipt_circuit(n_qubits, depth, p_measure, seed=None):
    """
    Build MIPT test circuit.
    
    Alternates entangling gates with probabilistic resets.
    TSO predicts phase transition at p_measure ≈ 0.31.
    """
    if seed is not None:
        np.random.seed(seed)
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    for _ in range(depth):
        # Random single-qubit rotations
        for i in range(n_qubits):
            qc.u(np.random.uniform(0, 2*np.pi),
                 np.random.uniform(0, 2*np.pi),
                 np.random.uniform(0, 2*np.pi), i)
        
        # Entangling layer (nearest-neighbor CNOTs)
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        
        # Probabilistic measurement (simulated via reset)
        for i in range(n_qubits):
            if np.random.random() < p_measure:
                qc.reset(i)
        
        qc.barrier()
    
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# =============================================================================
# ANALYSIS
# =============================================================================

def estimate_entropy(counts, n_qubits):
    """Estimate entropy from measurement counts."""
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    
    # Shannon entropy as proxy
    nonzero = probs > 0
    entropy = -np.sum(probs[nonzero] * np.log2(probs[nonzero]))
    normalized = entropy / n_qubits
    
    return normalized


def run_circuit(circuit, backend, shots):
    """Run circuit on backend, return counts."""
    if isinstance(backend, AerSimulator):
        tc = transpile(circuit, backend)
        result = backend.run(tc, shots=shots).result()
        return result.get_counts()
    else:
        # IBM Runtime
        tc = transpile(circuit, backend)
        sampler = SamplerV2(backend)
        job = sampler.run([tc], shots=shots)
        result = job.result()
        return result[0].data.meas.get_counts()


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_threshold(backend, shots=4096):
    """Test 1: Does transition occur at p_c ≈ 0.31?"""
    print("\n--- Test 1: Threshold Detection ---")
    
    p_values = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    entropies = []
    
    for p in p_values:
        # Average over multiple random circuits
        ent_samples = []
        for seed in range(5):
            circuit = build_mipt_circuit(6, 12, p, seed=seed)
            counts = run_circuit(circuit, backend, shots)
            ent_samples.append(estimate_entropy(counts, 6))
        
        avg_ent = np.mean(ent_samples)
        entropies.append(avg_ent)
        print(f"  p = {p:.2f}: entropy = {avg_ent:.3f}")
    
    entropies = np.array(entropies)
    
    # Find transition point (steepest descent)
    gradients = np.abs(np.diff(entropies) / np.diff(p_values))
    transition_idx = np.argmax(gradients)
    measured_pc = (p_values[transition_idx] + p_values[transition_idx + 1]) / 2
    
    passed = abs(measured_pc - P_C) < 0.05
    
    print(f"\n  TSO predicts: p_c = {P_C:.3f}")
    print(f"  Measured:     p_c ≈ {measured_pc:.3f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        'test': 'threshold',
        'predicted': P_C,
        'measured': measured_pc,
        'passed': passed,
        'p_values': p_values.tolist(),
        'entropies': entropies.tolist()
    }


def test_crossover_shape(backend, shots=4096):
    """Test 2: Is transition sigmoid (not linear)?"""
    print("\n--- Test 2: Crossover Shape ---")
    
    p_values = np.linspace(0.10, 0.50, 9)
    entropies = []
    
    for p in p_values:
        circuit = build_mipt_circuit(6, 12, p, seed=42)
        counts = run_circuit(circuit, backend, shots)
        entropies.append(estimate_entropy(counts, 6))
    
    entropies = np.array(entropies)
    
    # Fit TSO model (tanh)
    s_max = np.max(entropies)
    tso_pred = tso_entropy(p_values, s_max=s_max, kappa=1.35)
    tso_residual = np.sum((entropies - tso_pred)**2)
    
    # Fit linear model
    coeffs = np.polyfit(p_values, entropies, 1)
    linear_pred = np.polyval(coeffs, p_values)
    linear_residual = np.sum((entropies - linear_pred)**2)
    
    passed = tso_residual < linear_residual
    improvement = (linear_residual - tso_residual) / linear_residual * 100
    
    print(f"  TSO (tanh) residual: {tso_residual:.4f}")
    print(f"  Linear residual:     {linear_residual:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        'test': 'crossover_shape',
        'tso_residual': tso_residual,
        'linear_residual': linear_residual,
        'passed': passed
    }


def test_alpha_value(backend, shots=4096):
    """Test 3: Does α = 7/(1+e×ρ_c) match behavior?"""
    print("\n--- Test 3: Alpha Prediction ---")
    
    predicted_alpha = compute_alpha(0.5)
    print(f"  TSO predicts: α = {predicted_alpha:.3f}")
    
    # This is a simplified test - checking that the predicted
    # alpha gives sensible crystallization behavior
    
    qc = QuantumCircuit(4, 4)
    qc.h(0)  # Seed superposition
    
    angle = np.arctan(predicted_alpha) / 2
    for _ in range(5):
        for i in range(3):
            qc.rxx(angle, i, i+1)
            qc.ryy(angle, i, i+1)
    
    qc.measure([0,1,2,3], [0,1,2,3])
    
    counts = run_circuit(qc, backend, shots)
    entropy = estimate_entropy(counts, 4)
    
    # Check entropy is in expected range (not too high, not too low)
    passed = 0.3 < entropy < 0.8
    
    print(f"  Crystallization entropy: {entropy:.3f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        'test': 'alpha_value',
        'predicted_alpha': predicted_alpha,
        'entropy': entropy,
        'passed': passed
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TSO Validation Tests')
    parser.add_argument('--backend', default='aer_simulator',
                        help='aer_simulator or IBM backend name')
    parser.add_argument('--shots', type=int, default=4096,
                        help='Shots per circuit')
    parser.add_argument('--output', default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()
    
    print("=" * 50)
    print("TSO VALIDATION TESTS")
    print("=" * 50)
    print(f"Backend: {args.backend}")
    print(f"Shots: {args.shots}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Initialize backend
    if args.backend == 'aer_simulator':
        backend = AerSimulator()
    else:
        if not HAS_IBM_RUNTIME:
            print("ERROR: qiskit-ibm-runtime required for IBM backends")
            return
        service = QiskitRuntimeService()
        backend = service.backend(args.backend)
    
    # Run tests
    results = {
        'backend': args.backend,
        'shots': args.shots,
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    results['tests'].append(test_threshold(backend, args.shots))
    results['tests'].append(test_crossover_shape(backend, args.shots))
    results['tests'].append(test_alpha_value(backend, args.shots))
    
    # Summary
    passed = sum(t['passed'] for t in results['tests'])
    total = len(results['tests'])
    
    print("\n" + "=" * 50)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 50)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return results


if __name__ == '__main__':
    main()
