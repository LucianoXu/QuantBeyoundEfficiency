from typing import Any
from pathlib import Path

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

    print(" >> Experiment Top-level")

    if isinstance(args, (str, Path)):
        print(" >> Load configurations from", args)
        args = load_yaml_config(args)


    if args["config_type"] == "model":
        return model_factory(args)
    
    if args["config_type"] == "bench":
        return bench_factory(args)
    
    if args["config_type"] == "model_bench":
        return run_model_bench(args)
    
    if args["config_type"] == "model_bench_matrix":
        return run_model_bench_matrix(args)
    
    else:
        raise ValueError("Invalid Config Type.")


def run_model_bench(args: dict[str, Any], tokenizer: Any = None, model: Any = None):

    output_dir = Path(args['output_dir'])
    # create a folder, and raise an error if it already exists
    output_dir.mkdir(parents=True, exist_ok=False)

    # mirror all console output of this run into output_dir/log.txt
    with tee_console(output_dir / "log.txt"):

        print(" >> Model-Benchmark Pair Experiment")

        seed_everything(args['random_seed'])

        model_args = args['model']
        model_args['config_type'] = "model"
        if model is None or tokenizer is None: 
            tokenizer, model = model_factory(model_args)

        bench_args = args['bench']
        bench_args['config_type'] = "bench"
        # propagate the run-level seed so the bench can pass it to lm_eval
        # (lm_eval re-seeds internally and otherwise ignores our global seeding)
        bench_args['random_seed'] = args['random_seed']

        bench = bench_factory(bench_args)

        # write the config and the environment snapshot to the folder
        save_yaml_config(args, output_dir / "config.yaml")
        save_json(collect_environment(), output_dir / "env.json")

        res = bench.eval(tokenizer, model, output_dir)

    return res

def run_model_bench_matrix(args: dict[str, Any]):

    output_dir = Path(args['output_dir'])
    # create a folder, and raise an error if it already exists
    output_dir.mkdir(parents=True, exist_ok=False)
    # mirror all console output of this run into output_dir/log.txt
    with tee_console(output_dir / "log.txt"):

        print(" >> Model-Benchmark Matrix Experiment")

        seed_everything(args['random_seed'])

        model_args_ls = args['models']
        bench_args_ls = args['benches']

        # record the environment for the matrix root (each pair also gets its own)
        save_json(collect_environment(), output_dir / "env.json")

        # check whether all model ids and bench ids are unique
        model_ids = [m["id"] for m in model_args_ls]
        bench_ids = [b["id"] for b in bench_args_ls]
        for kind, ids in (("model", model_ids), ("bench", bench_ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            if duplicates:
                raise ValueError(
                    f"Duplicate {kind} id(s) {duplicates}: each {kind} id must be unique "
                    f"(ids form the per-run output folder name '<model_id>-<bench_id>')."
                )

        # dispatch all works
        for model_args in model_args_ls:

            tokenizer, model = model_factory(model_args)
            model_args = model_args.copy()
            model_id = model_args["id"]
            del model_args["id"]

            for bench_args in bench_args_ls:

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