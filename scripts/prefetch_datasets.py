"""
Downloads all benchmark datasets into the Huggingface cache to avoid race conditions during parallel evaluation.

This step is not necessary at single process evaluation.
"""

from src.benches.factory import LM_EVAL_TASKS


def prefetch_datasets() -> None:
    """ Loads all lm-evaluation-harness tasks to load download datasets into the huggingface cache. """
    bench_names = set(LM_EVAL_TASKS)

    task_ids = sorted({LM_EVAL_TASKS[n] for n in bench_names})
    print(f" >> Prefetching {len(task_ids)} task group(s) into the shared HF cache: {task_ids}")

    from lm_eval.tasks import TaskManager

    tm = TaskManager()
    failures: list[tuple[str, str]] = []
    for tid in task_ids:
        print(f" >> loading (downloads + generates dataset): {tid}", flush=True)
        try:
            tm.load(tid)
            print(f" >> done: {tid}", flush=True)
        except Exception as e:
            print(f" !! FAILED to prefetch {tid}: {e!r}", flush=True)
            failures.append((tid, repr(e)))

    if failures:
        print("\n >> Some datasets failed to prefetch:")
        for tid, err in failures:
            print(f"    - {tid}: {err}")
        exit(1)

    print("\n >> All datasets prefetched. Safe to submit the parallel array job.")

if __name__ == "__main__":
    prefetch_datasets()