"""

re-download all benchmark datasets into the shared HuggingFace cache.

This step avoids data racing for concurrent evaluation. For single process it is not necessary.

python scripts/prefetch_datasets.py

"""

from src.benches.factory import LM_EVAL_TASKS


if __name__ == "__main__":

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