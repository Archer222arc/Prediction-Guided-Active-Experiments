#!/usr/bin/env python3
import json
import sys

def analyze_ci_results(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Extract PGAE and Adaptive_PGAE results
    pgae_results = []
    adaptive_pgae_results = []
    
    for method_name, method_data in results.items():
        if method_name == 'PGAE':
            # Get trials from the method data
            trials = method_data.get('trials', [])
            for trial in trials:
                pgae_results.append({
                    'avg_ci_length': trial['avg_ci_length'],
                    'coverage_rate': trial['results_snapshot']['coverage_rate'],
                    'mse': trial['results_snapshot']['mse'],
                    'internal_preset': trial.get('internal_preset', {}),
                    'method': 'PGAE'
                })
        elif method_name == 'Adaptive_PGAE':
            # For Adaptive_PGAE, the parameters are at the top level
            adaptive_pgae_results.append({
                'avg_ci_length': method_data['avg_ci_length'],
                'coverage_rate': method_data['results_snapshot']['coverage_rate'],
                'mse': method_data['results_snapshot']['mse'],
                'gamma': method_data.get('gamma'),
                'batch_size': method_data.get('batch_size'),
                'design_update_freq': method_data.get('design_update_freq'),
                'warmup_batches': method_data.get('warmup_batches'),
                'method': 'Adaptive_PGAE'
            })
    
    # Find best CI performance for each method
    def get_best_ci(results_list, method_name):
        if not results_list:
            return None
        
        # Sort by CI length (smaller is better) and coverage rate (95% is ideal)
        best_result = min(results_list, key=lambda x: (x['avg_ci_length'], abs(x['coverage_rate'] - 0.95)))
        
        print(f"\n=== 最佳 {method_name} 参数 ===")
        print(f"平均CI宽度: {best_result['avg_ci_length']:.6f}")
        print(f"覆盖率: {best_result['coverage_rate']:.2f}")
        print(f"MSE: {best_result['mse']:.6f}")
        
        if method_name == 'PGAE':
            print(f"内部参数设置: {best_result['internal_preset']}")
        else:  # Adaptive_PGAE
            print(f"参数设置:")
            print(f"  gamma: {best_result.get('gamma')}")
            print(f"  batch_size: {best_result.get('batch_size')}")
            print(f"  design_update_freq: {best_result.get('design_update_freq')}")
            print(f"  warmup_batches: {best_result.get('warmup_batches')}")
        
        return best_result
    
    pgae_best = get_best_ci(pgae_results, "PGAE")
    adaptive_pgae_best = get_best_ci(adaptive_pgae_results, "Adaptive_PGAE")
    
    # Compare the two
    if pgae_best and adaptive_pgae_best:
        print(f"\n=== 比较结果 ===")
        print(f"PGAE最佳CI宽度: {pgae_best['avg_ci_length']:.6f}")
        print(f"Adaptive_PGAE最佳CI宽度: {adaptive_pgae_best['avg_ci_length']:.6f}")
        
        if pgae_best['avg_ci_length'] < adaptive_pgae_best['avg_ci_length']:
            print("PGAE表现更好 (CI更窄)")
        else:
            print("Adaptive_PGAE表现更好 (CI更窄)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_ci_results.py <json_file>")
        sys.exit(1)
    
    analyze_ci_results(sys.argv[1])