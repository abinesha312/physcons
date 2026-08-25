"""
CLI for physcons demonstrations.

Entry point: `physcons demo`
"""

import argparse
import sys
import numpy as np

from .data import generate_conserved_field, generate_broken_field
from .router import PatchRouter
from .conservation import ConservationLayer


def demo_command():
    """
    Run the conservation demo.
    
    This demonstrates:
    1. A conserved field remains conserved after routing
    2. A broken field is detected and corrected
    3. The Rust kernel is working (or Python fallback is used)
    """
    print("=" * 70)
    print("PhysCons Demo: Patch Routing + Hard Conservation")
    print("=" * 70)
    print()
    
    # Initialize components
    print("Initializing router and conservation layer...")
    router = PatchRouter(patch_size=10, n_experts=3, seed=42)
    conservation = ConservationLayer(threshold=1e-3)
    
    if conservation.is_rust_available:
        print("✓ Rust extension loaded")
    else:
        print("⚠ Rust extension not available, using Python fallback")
        error = conservation.get_import_error()
        if error:
            print(f"  Import error: {error}")
    print()
    
    # Demo 1: Conserved field
    print("-" * 70)
    print("Demo 1: Field that starts conserved")
    print("-" * 70)
    
    field1, target1 = generate_conserved_field(n_points=100, total_mass=1000.0)
    print(f"Initial field sum: {field1.sum():.6f}")
    print(f"Target sum:        {target1:.6f}")
    print(f"Initial residual:  {abs(field1.sum() - target1):.6e}")
    print()
    
    # Route through patches
    routed1 = router.forward(field1)
    print(f"After routing sum: {routed1.sum():.6f}")
    
    # Apply conservation
    corrected1, diag1 = conservation.apply_conservation(routed1, target1)
    print(f"Residual:          {diag1['residual']:.6e}")
    print(f"Relative residual: {diag1['relative_residual']:.6e}")
    print(f"Corrected:         {diag1['corrected']}")
    
    if diag1['corrected']:
        print(f"After correction:  {corrected1.sum():.6f}")
        print(f"Final residual:    {diag1['residual_after']:.6e}")
    print()
    
    # Demo 2: Broken field
    print("-" * 70)
    print("Demo 2: Field that violates conservation")
    print("-" * 70)
    
    field2, target2 = generate_broken_field(
        n_points=100,
        total_mass=1000.0,
        violation_fraction=0.1
    )
    print(f"Initial field sum: {field2.sum():.6f}")
    print(f"Target sum:        {target2:.6f}")
    print(f"Initial residual:  {abs(field2.sum() - target2):.6e}")
    print()
    
    # Route through patches
    routed2 = router.forward(field2)
    print(f"After routing sum: {routed2.sum():.6f}")
    
    # Apply conservation
    corrected2, diag2 = conservation.apply_conservation(routed2, target2)
    print(f"Residual:          {diag2['residual']:.6e}")
    print(f"Relative residual: {diag2['relative_residual']:.6e}")
    print(f"Corrected:         {diag2['corrected']}")
    
    if diag2['corrected']:
        print(f"After correction:  {corrected2.sum():.6f}")
        print(f"Final residual:    {diag2['residual_after']:.6e}")
    print()
    
    # Summary
    print("=" * 70)
    print("Demo complete.")
    print()
    print("What this shows:")
    print("  • Patch-based routing (Shodh-MoE style)")
    print("  • Conservation residual computation (PI-MFM style)")
    print("  • Hard constraint correction (PI-HC-MoE style)")
    print()
    print("Limitations:")
    print("  • Toy 1D field only (not 2D/3D PDEs)")
    print("  • Simple linear experts (not foundation models)")
    print("  • No real physics simulation")
    print("  • Not validated on real-world data")
    print("=" * 70)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="physcons",
        description="Toy PoC: patch routing with hard conservation"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    subparsers.add_parser("demo", help="Run conservation demo")
    
    args = parser.parse_args()
    
    if args.command == "demo":
        demo_command()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
