# RAP: Risk-Aware Prediction

RAP is a PyTorch-based machine learning project for trajectory forecasting that biases predictions towards risk to help planners estimate and react to dangerous but low-probability events. The project uses a Conditional Variational Auto-Encoder (CVAE) model for pedestrian trajectory prediction.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Installation
Create a virtual environment and install dependencies:
- `python3 -m venv venv`
- `source venv/bin/activate`
- `bash install.sh` -- installs pip==23.0, PyTorch 1.13.1+cu117, and all requirements. Takes 10-15 minutes. NEVER CANCEL. Set timeout to 30+ minutes.

**CRITICAL**: Installation may fail due to network connectivity issues or externally-managed Python environments. If installation fails:
- Try using `pip install --user` for user-level installation
- Or use system packages where available: `apt-get install python3-pytorch python3-matplotlib python3-numpy`
- Installation failure is common in restricted environments due to PyPI connectivity issues

### Configuration Setup
Before running any training:
- Edit `risk_biased/config/paths.py` to set your data paths and log directory
- For WOMD: Set `base_path` to your Waymo dataset location  
- For didactic simulation: Paths are auto-generated, no setup required

### Building and Testing
- **Package Structure**: Use `PYTHONPATH=/home/runner/work/RAP/RAP python3 -c "import risk_biased"` to test package accessibility
- **Run Tests**: `python3 -m pytest tests/` -- takes 5-10 minutes. NEVER CANCEL. Set timeout to 20+ minutes.
  - Note: Requires pytest installation. If unavailable, use: `PYTHONPATH=/home/runner/work/RAP/RAP python3 tests/runtests.py`
- **Code Formatting**: `python3 -m yapf --style pep8 -r risk_biased/` (yapf==0.32.0 required)

### Running Training
Choose between two training scenarios:

**Didactic Simulation** (simpler, for testing):
- `PYTHONPATH=/home/runner/work/RAP/RAP python3 scripts/train_didactic.py`
- Takes 30-60 minutes for full training. NEVER CANCEL. Set timeout to 90+ minutes.
- Dataset auto-generated with parameters in `risk_biased/config/learning_config.py`

**Waymo Open Motion Dataset** (production):
- First preprocess data: `python3 scripts/scripts_utils/generate_dataset_waymo.py <waymo_path>/scenario/validation <waymo_path>/interactive_veh_type/sample --num_parallel=16 --debug_size=1000`
- `PYTHONPATH=/home/runner/work/RAP/RAP python3 scripts/train_interaction.py`
- Takes 2-4 hours for full training. NEVER CANCEL. Set timeout to 300+ minutes.

### Running Evaluation and Visualization
- **Interactive Interface**: `python3 scripts/scripts_utils/plotly_interface.py --load_from=<checkpoint.ckpt> --cfg_path=<config_path>`
- **Evaluation Scripts**: Multiple scripts in `scripts/eval_scripts/` for computing stats, plotting, etc.

## Validation

### Always Run These Validation Steps
- Test package structure: `PYTHONPATH=/home/runner/work/RAP/RAP python3 -c "import risk_biased; print('Success: Package accessible')"`
- Check directory structure matches expected layout: `ls -la` (should show risk_biased/, scripts/, tests/, etc.)
- Validate configuration files exist: `ls risk_biased/config/` (should show learning_config.py, waymo_config.py, paths.py, planning_config.py)

### Before Making Code Changes
- **For training changes**: Test at least one training step of didactic simulation to verify model pipeline works
- **For model changes**: Run targeted tests like `python3 -m pytest tests/risk_biased/models/test_biased_cvae_model.py -v` (if pytest available)
- **For config changes**: Validate configs load without errors: `PYTHONPATH=/home/runner/work/RAP/RAP python3 -c "from risk_biased.config import learning_config; print('Config loaded')"`

### User Scenarios to Test After Changes

**Scenario 1 - Basic Setup Validation**:
1. Clone fresh repository
2. Run `PYTHONPATH=/home/runner/work/RAP/RAP python3 -c "import risk_biased; print('Package accessible')"`
3. Check `ls risk_biased/config/` shows all 4 config files
4. Verify `python3 -c "from risk_biased.config import learning_config; print('Config loaded')"`

**Scenario 2 - Training Pipeline (Didactic)**:
1. Complete basic setup validation
2. Create virtual environment: `python3 -m venv venv && source venv/bin/activate`
3. Attempt installation: `bash install.sh` (may fail due to network issues)
4. If successful, run: `python3 scripts/train_didactic.py --num_epochs_cvae=1 --num_epochs_bias=1` for quick test

**Scenario 3 - Code Quality**:
1. Make code changes to risk_biased package
2. Check syntax: `python3 -m py_compile risk_biased/models/*.py`
3. If yapf available: `yapf --style pep8 --diff risk_biased/` to check formatting
4. Test imports still work after changes

## Common Issues and Limitations

### Installation Problems
- **Network Timeouts**: PyPI downloads may timeout. Increase pip timeout with `--timeout=300`
- **CUDA Dependencies**: PyTorch with CUDA support requires specific versions. Use exact version from requirements.txt
- **Missing Dependencies**: mmcv, pytorch-lightning versions must match exactly. Use `pip install mmcv==1.4.7 pytorch-lightning==1.7.7`

### Environment Limitations  
- **Externally-managed Python**: Some environments prevent system-wide pip installs. Always use virtual environments
- **CUDA Availability**: Training requires CUDA-capable GPU. CPU training is extremely slow (hours -> days)
- **Memory Requirements**: Risk estimation with many Monte Carlo samples may cause GPU OOM. Reduce `n_mc_samples_biased` in config if needed

### Training Limitations
- **Two-Phase Training**: First trains unbiased predictor, then biased encoder. Second phase may be slow due to Monte Carlo sampling
- **WandB Login**: May require `wandb login` for experiment tracking
- **Checkpoint Loading**: Use `--load_from "<wandb_id>"` to resume from WandB checkpoint

## Project Structure

### Key Directories
- `risk_biased/`: Main Python package
  - `config/`: Configuration files for different scenarios
  - `models/`: CVAE model implementations  
  - `predictors/`: Trajectory prediction logic
  - `scene_dataset/`: Data loading and preprocessing
  - `utils/`: Utility functions for loss, risk estimation, etc.
- `scripts/`: Training and evaluation scripts
  - `train_didactic.py`: Simple simulation training
  - `train_interaction.py`: Waymo dataset training  
  - `eval_scripts/`: Visualization and analysis tools
- `tests/`: Comprehensive test suite with pytest

### Configuration Files
- `learning_config.py`: Didactic simulation parameters
- `waymo_config.py`: Waymo dataset parameters  
- `paths.py`: Data and log directory paths (MUST be configured)
- `planning_config.py`: MPC planner parameters

### Key Scripts Output
```bash
# Repository root structure
ls -la /home/runner/work/RAP/RAP/
total 80
drwxr-xr-x 10 runner docker  4096 .
drwxr-xr-x  3 runner docker  4096 ..
-rw-r--r--  1 runner docker 14529 LICENSE.md
-rw-r--r--  1 runner docker  5401 README.md
drwxr-xr-x  2 runner docker  4096 image
-rw-r--r--  1 runner docker   160 install.sh
drwxr-xr-x  2 runner docker  4096 notebooks
-rw-r--r--  1 runner docker   344 requirements.txt
drwxr-xr-x  9 runner docker  4096 risk_biased
drwxr-xr-x  4 runner docker  4096 scripts
-rw-r--r--  1 runner docker   690 setup.py
drwxr-xr-x  3 runner docker  4096 tests

# Package structure  
ls risk_biased/
__init__.py  config  models  mpc_planner  predictors  scene_dataset  utils

# Configuration files
ls risk_biased/config/  
learning_config.py  paths.py  planning_config.py  waymo_config.py

# Test structure
ls tests/risk_biased/
models  mpc_planner  predictors  scene_dataset  utils

# Install script contents
cat install.sh
pip install pip==23.0
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install -e .
```

### Dependencies Overview
Key packages from requirements.txt:
- torch==1.13.1+cu117 (must install separately with CUDA index)
- pytorch-lightning==1.7.7  
- mmcv==1.4.7
- numpy==1.26.4
- wandb (for experiment tracking)
- plotly (for interactive visualization)
- pytest (for testing)
- yapf==0.32.0 (for code formatting)

Always check exact versions in requirements.txt as ML libraries are version-sensitive.