[Report Sketch](https://github.com/LucianoXu/QuantBeyoundEfficiency/wiki/Report) | [Devlog](./devlog.md)

# Quant Beyound Efficiency

## Setup

1. copy `.env.example` to `.env` and paste your HuggingFace token there. (https://huggingface.co/settings/tokens).
2. Create a python environment according to `requirements.txt`.
3. Go through `cookbook.ipynb`.

## Project Structure

- `README.md`: this file, project introduction
- `requirements.txt`: python environment requirements
- `references/`: documents like papers and project notes
- `report/`: the final latex report
- `results/`: important experiment results (tracked)
- `trials/`: intermediate experiment results (gitignored)
- `src/`: python source files
- `scripts/`: script files for all jobs
- `templates/`: task configuration templates
- `.env`: environment variable configuration (gitignored)
- `devlog.md`: development and progress log
- `cookbook.ipynb`: push-button experiment cookbooks

## Scope

- Model size up to 4B.
- Only compare HuggingFace/Pytorch implementations. No optimization by compilation / vLLM / kernel fusion for now.

## Dependency

- `transformers`
- `lm-evaluation-harness`: the standard benchmarking library with rich tunable parameters.


## Configuration Types

- **model**
- **model_bench**
- **model_bench_matrix**

## Guidelines

- Task parameters are tracked by `.yaml` configuration files. Use factories for model and benchmark construction.
- New models, quantization techniques, benchmarks and analysis follow the same interface and extend the factory.
- Raw experiment data are preserved in `trials/` by default. Move important results into `results/`, which is tracked and synced.



[ ] Refine the scope.

[ ] Current interface may need adoption for interpretability analysis.
