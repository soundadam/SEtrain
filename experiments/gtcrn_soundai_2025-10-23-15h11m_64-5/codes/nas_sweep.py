#!/usr/bin/env python3
"""
Neural Architecture Search (NAS) Parameter Sweep Script
Systematically evaluates different architecture configurations for GTCRN model
"""

import os
import sys
import argparse
import itertools
import subprocess
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
import yaml


class NASSweepRunner:
    """Orchestrates parameter sweep experiments for NAS"""

    def __init__(self, base_config_path, nas_config_path, output_dir='experiments/nas_sweep'):
        self.base_config_path = base_config_path
        self.nas_config_path = nas_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load base configurations
        self.base_config = OmegaConf.load(base_config_path)
        self.nas_config = OmegaConf.load(nas_config_path)

        # Parameter search space
        self.param_space = {
            'channelsize': [32, 64, 96, 128],
            'gt_blocks_repeat': [3, 4, 5, 6]
        }

        # Results tracking
        self.results = []
        self.results_file = self.output_dir / 'sweep_results.yaml'

    def generate_config_for_params(self, params_dict):
        """Generate a temporary config file for specific parameter values"""
        config = OmegaConf.to_container(self.base_config, resolve=True)
        nas_config = OmegaConf.to_container(self.nas_config, resolve=True)

        # Update NAS parameters
        nas_config['nas_config'].update(params_dict)

        # Merge configs
        config.update(nas_config)

        # Create unique experiment name
        param_str = '_'.join([f"{k}{v}" for k, v in params_dict.items()])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"nas_{param_str}_{timestamp}"

        config['trainer']['exp_path'] = str(self.output_dir / exp_name)

        return config, exp_name

    def save_temp_config(self, config, exp_name):
        """Save temporary configuration file"""
        temp_config_path = self.output_dir / f"{exp_name}_config.yaml"
        OmegaConf.save(config, temp_config_path)
        return temp_config_path

    def run_single_experiment(self, params_dict, gpu_device='0'):
        """Execute a single training experiment with specified parameters"""
        print(f"\n{'='*80}")
        print(f"Starting experiment with parameters: {params_dict}")
        print(f"{'='*80}\n")

        config, exp_name = self.generate_config_for_params(params_dict)
        temp_config_path = self.save_temp_config(config, exp_name)

        # Prepare environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_device

        # Run training
        cmd = [
            sys.executable,
            'train.py',
            '--config', str(temp_config_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse results from experiment directory
            exp_dir = Path(config['trainer']['exp_path'])
            best_model_files = list(exp_dir.glob('checkpoints/best_model_*.tar'))

            if best_model_files:
                # Extract epoch number from best model filename
                best_epoch = int(best_model_files[0].stem.split('_')[-1])
            else:
                best_epoch = None

            experiment_result = {
                'params': params_dict,
                'exp_name': exp_name,
                'exp_path': str(exp_dir),
                'best_epoch': best_epoch,
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }

        except subprocess.CalledProcessError as e:
            print(f"Experiment failed with error:\n{e.stderr}")
            experiment_result = {
                'params': params_dict,
                'exp_name': exp_name,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

        # Save results incrementally
        self.results.append(experiment_result)
        self.save_results()

        return experiment_result

    def save_results(self):
        """Save sweep results to YAML file"""
        with open(self.results_file, 'w') as f:
            yaml.dump({
                'sweep_results': self.results,
                'param_space': self.param_space
            }, f, default_flow_style=False)

    def sweep_parameter(self, param_name, gpu_device='0'):
        """Sweep a single parameter while keeping others at default"""
        if param_name not in self.param_space:
            raise ValueError(f"Unknown parameter: {param_name}")

        print(f"\n{'#'*80}")
        print(f"# Sweeping parameter: {param_name}")
        print(f"# Values: {self.param_space[param_name]}")
        print(f"{'#'*80}\n")

        # Get default values from NAS config
        default_params = {
            'channelsize': self.nas_config.nas_config.channelsize,
            'gt_blocks_repeat': self.nas_config.nas_config.gt_blocks_repeat
        }

        for value in self.param_space[param_name]:
            params = default_params.copy()
            params[param_name] = value

            self.run_single_experiment(params, gpu_device)

    def sweep_grid(self, gpu_device='0'):
        """Perform full grid search over all parameters"""
        print(f"\n{'#'*80}")
        print(f"# Full Grid Search")
        print(f"# Total experiments: {len(self.param_space['channelsize']) * len(self.param_space['gt_blocks_repeat'])}")
        print(f"{'#'*80}\n")

        for channelsize in self.param_space['channelsize']:
            for gt_blocks_repeat in self.param_space['gt_blocks_repeat']:
                params = {
                    'channelsize': channelsize,
                    'gt_blocks_repeat': gt_blocks_repeat
                }
                self.run_single_experiment(params, gpu_device)

    def analyze_results(self):
        """Analyze and summarize sweep results"""
        if not self.results:
            print("No results to analyze")
            return

        print(f"\n{'='*80}")
        print("SWEEP RESULTS SUMMARY")
        print(f"{'='*80}\n")

        completed = [r for r in self.results if r['status'] == 'completed']
        failed = [r for r in self.results if r['status'] == 'failed']

        print(f"Total experiments: {len(self.results)}")
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}\n")

        if completed:
            print("Completed Experiments:")
            print(f"{'Params':<40} {'Best Epoch':<15} {'Exp Path'}")
            print("-" * 100)
            for r in completed:
                params_str = str(r['params'])
                print(f"{params_str:<40} {str(r.get('best_epoch', 'N/A')):<15} {r['exp_path']}")

        if failed:
            print("\nFailed Experiments:")
            for r in failed:
                print(f"  {r['params']}: {r.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(
        description='NAS Parameter Sweep for GTCRN Audio Enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sweep channelsize parameter only
  python nas_sweep.py --sweep-param channelsize --gpu 0

  # Sweep gt_blocks_repeat parameter only
  python nas_sweep.py --sweep-param gt_blocks_repeat --gpu 0

  # Full grid search
  python nas_sweep.py --grid --gpu 0,1

  # Analyze existing results
  python nas_sweep.py --analyze-only
        """
    )

    parser.add_argument(
        '--config',
        default='configs/cfg_train.yaml',
        help='Path to base training config'
    )

    parser.add_argument(
        '--nas-config',
        default='configs/nas_train.yaml',
        help='Path to NAS config'
    )

    parser.add_argument(
        '--sweep-param',
        choices=['channelsize', 'gt_blocks_repeat'],
        help='Parameter to sweep (one at a time)'
    )

    parser.add_argument(
        '--grid',
        action='store_true',
        help='Perform full grid search'
    )

    parser.add_argument(
        '--gpu',
        default='0',
        help='GPU device(s) to use (e.g., "0" or "0,1")'
    )

    parser.add_argument(
        '--output-dir',
        default='experiments/nas_sweep',
        help='Output directory for sweep results'
    )

    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze existing results without running experiments'
    )

    args = parser.parse_args()

    # Initialize sweep runner
    runner = NASSweepRunner(
        base_config_path=args.config,
        nas_config_path=args.nas_config,
        output_dir=args.output_dir
    )

    if args.analyze_only:
        # Load existing results
        if runner.results_file.exists():
            with open(runner.results_file) as f:
                data = yaml.safe_load(f)
                runner.results = data.get('sweep_results', [])
        runner.analyze_results()

    elif args.sweep_param:
        # Single parameter sweep
        runner.sweep_parameter(args.sweep_param, args.gpu)
        runner.analyze_results()

    elif args.grid:
        # Full grid search
        runner.sweep_grid(args.gpu)
        runner.analyze_results()

    else:
        parser.print_help()
        print("\nError: Must specify --sweep-param, --grid, or --analyze-only")
        sys.exit(1)


if __name__ == '__main__':
    main()
