# Templates

This directory contains YAML configuration files used to define the experiments.
All experiments were either passive benchmark evaluations across different quantization levels or an active adversarial attack between full precision and a quantization level.

## Overview of Experiment Templates

Configuration fall into one of two categories defined by the `config_type` argument:
* `config_type: attack` is used for evaluation the GCG attack against target models
* `config_type: model_bench_matrix` Evaluates different benchmarks across different quantization levels

## How to create a new configuration

Copy one of the minimal examples at `templates/test` and compare with the detailed options in 'templates\experiments'
