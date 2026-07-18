from typing import Any
from pathlib import Path
import torch
import gc

from .utils import (
    load_yaml_config,
    save_yaml_config,
    save_json,
    seed_everything,
    tee_console,
    collect_environment,
)

from .models.factory import model_factory
from .benches.factory import bench_factory


def run(args: dict[str, Any] | str | Path) -> Any:
    """ Initializes the experiment by loading the model and the benchmarking type/attack given by the config.

    Args:
        args: A dictionary holding the config arguments or a str/Path pointing to the .yaml configuration.

    Returns:
        The results of the initialized experiment.

    Raises:
        ValueError: If the extracted 'config_type' does not map to a supported experiment.
    """

    print(" >> Experiment Top-level")

    if isinstance(args, (str, Path)):
        print(" >> Load configurations from", args)
        args = load_yaml_config(args)


    if args["config_type"] == "model":
        return model_factory(args, seed=args.get("random_seed"))
    
    elif args["config_type"] == "bench":
        return bench_factory(args)
    
    elif args["config_type"] == "model_bench":
        return run_model_bench(args)
    
    elif args["config_type"] == "model_bench_matrix":
        return run_model_bench_matrix(args)

    elif args["config_type"] == "attack":
        from .attacks.factory import attack_factory
        return attack_factory(args)

    else:
        raise ValueError("Invalid Config Type.")


def run_model_bench(args: dict[str, Any], tokenizer: Any = None, model: Any = None) -> dict[str, Any] | None:
    """ Executes a single evaluation run for a given model and a given benchmark.

    Args:
        args: Configuration dictionary for the execution details.
        tokenizer: Optional pre-loaded tokenizer for target model. Defaults to None.
        model: Optional target model. Defaults to None.

    Returns:
        A dictionarz containing the compiled evaluation results or None if the benchmark throws an exception.

    Raises:
         FileExistsError: If the output tracking folder already exists. (e.g. trials)
    """

    model_not_passed = model is None or tokenizer is None
    output_dir = Path(args['output_dir'])
    # create a folder, and raise an error if it already exists
    output_dir.mkdir(parents=True, exist_ok=False)

    # mirror all console output of this run into output_dir/log.txt
    with tee_console(output_dir / "log.txt"):

        print(" >> Model-Benchmark Pair Experiment")

        seed_everything(args['random_seed'])

        model_args = args['model']
        model_args['config_type'] = "model"
        if model_not_passed: 
            tokenizer, model = model_factory(model_args, seed=args['random_seed'])

        bench_args = args['bench']
        bench_args['config_type'] = "bench"
        # propagate the run-level seed so the bench can pass it to lm_eval
        # (lm_eval re-seeds internally and otherwise ignores our global seeding)
        bench_args['random_seed'] = args['random_seed']

        bench = bench_factory(bench_args)

        # write the config and the environment snapshot to the folder
        save_yaml_config(args, output_dir / "config.yaml")
        save_json(collect_environment(), output_dir / "env.json")

        res = None
        try: 
            res = bench.eval(tokenizer, model, output_dir)
        except Exception as e: 
            print(f"ERROR: Benchmark {bench_args} failed with error: {e}")

        if model_not_passed:
            del tokenizer, model
            gc.collect()
            torch.cuda.empty_cache()

    return res

def run_model_bench_matrix(args: dict[str, Any]):
    """ Executes a cross evaluation matrix for a multiple quantized models and benchmarks.

    Args:
        args: Configuration dictionary for the execution details.

    Raises:
         FileExistsError: If the output tracking folder already exists. (e.g. trials)
         ValueError: If duplicate model or benchmark ids are found.
    """

    output_dir = Path(args['output_dir'])
    # create a folder, and raise an error if it already exists
    output_dir.mkdir(parents=True, exist_ok=False)
    # mirror all console output of this run into output_dir/log.txt
    with tee_console(output_dir / "log.txt"):

        print(" >> Model-Benchmark Matrix Experiment")

        seed_everything(args['random_seed'])

        model_args_ls = args['models']

        # record the environment for the matrix root (each pair also gets its own)
        save_json(collect_environment(), output_dir / "env.json")

        # check whether all model ids and bench ids are unique
        model_ids = [m["id"] for m in model_args_ls]
        model_duplicates = sorted({i for i in model_ids if model_ids.count(i) > 1})
        if model_duplicates:
            raise ValueError(
                f"Duplicate model id(s) {model_duplicates}: each model id must be unique "
                f"(ids form the per-run output folder name '<model_id>-<bench_id>')."
            )

        # dispatch all works
        for model_args in model_args_ls:
            model_args = model_args.copy()
            model_id = model_args["id"]
            bench_args_ls = model_args.pop("benches")

            # check whether all model ids and bench ids are unique
            bench_ids = [b["id"] for b in bench_args_ls]
            bench_duplicates = sorted({i for i in bench_ids if bench_ids.count(i) > 1})
            if bench_duplicates:
                raise ValueError(
                    f"Duplicate bench id(s) {bench_duplicates} from model {model_id}: each bench id under a model must be unique "
                )

            print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Loading Model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ")
            tokenizer, model = model_factory(model_args, seed=args['random_seed'])


            for bench_args in bench_args_ls:
                print(" >>>>>>>>>>>>>>>>>>>>>>>>>> Loading Benchmark >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ")
                bench_args = bench_args.copy()
                bench_id = bench_args["id"]
                del bench_args["id"]

                model_bench_config = {
                    "config_type" : "model_bench",
                    "random_seed" : args['random_seed'],
                    "output_dir" : str(output_dir / f"{model_id}-{bench_id}"),
                    "model" : model_args,
                    "bench" : bench_args
                }
                run_model_bench(model_bench_config, tokenizer, model)
            del tokenizer, model  
            gc.collect()
            torch.cuda.empty_cache()