from __future__ import annotations

from typing import TYPE_CHECKING

_POLICY_EVAL_EXPORTS = {
    "DepthSummary",
    "EpisodeResult",
    "PolicyEvaluationConfig",
    "evaluate_policy_on_env",
    "evaluate_policy_on_scrambles",
    "summarize_results",
}

_COMPARE_EXPORTS = {
    "ModelSpec",
    "NamedPolicy",
    "compare_loaded_policies",
    "load_named_policies",
    "parse_model_spec",
    "parse_model_specs",
    "rank_policy_reports",
}

_SEARCH_EVAL_EXPORTS = {
    "BenchmarkCase",
    "BenchmarkEpisodeResult",
    "BenchmarkSummary",
    "SearchBenchmarkConfig",
    "evaluate_greedy_on_cases",
    "evaluate_search_on_cases",
    "generate_benchmark_cases",
    "rank_strategy_reports",
    "run_search_benchmark",
    "summarize_benchmark_results",
}

_BENCHMARK_SUITE_EXPORTS = {
    "SUITE_PRESETS",
    "WeightedAStarBenchmarkConfig",
    "generate_suite_cases",
    "load_benchmark_cases",
    "preset_config",
    "run_benchmark_suite",
    "run_davi_benchmark",
    "write_benchmark_cases",
}

if TYPE_CHECKING:
    from .compare_policies import (
        ModelSpec,
        NamedPolicy,
        compare_loaded_policies,
        load_named_policies,
        parse_model_spec,
        parse_model_specs,
        rank_policy_reports,
    )
    from .policy_eval import (
        DepthSummary,
        EpisodeResult,
        PolicyEvaluationConfig,
        evaluate_policy_on_env,
        evaluate_policy_on_scrambles,
        summarize_results,
    )
    from .search_eval import (
        BenchmarkCase,
        BenchmarkEpisodeResult,
        BenchmarkSummary,
        SearchBenchmarkConfig,
        evaluate_greedy_on_cases,
        evaluate_search_on_cases,
        generate_benchmark_cases,
        rank_strategy_reports,
        run_search_benchmark,
        summarize_benchmark_results,
    )
    from .benchmark_suite import (
        SUITE_PRESETS,
        WeightedAStarBenchmarkConfig,
        generate_suite_cases,
        load_benchmark_cases,
        preset_config,
        run_benchmark_suite,
        run_davi_benchmark,
        write_benchmark_cases,
    )

__all__ = sorted(
    _POLICY_EVAL_EXPORTS
    | _COMPARE_EXPORTS
    | _SEARCH_EVAL_EXPORTS
    | _BENCHMARK_SUITE_EXPORTS
)


def __getattr__(name: str):
    if name in _POLICY_EVAL_EXPORTS:
        from . import policy_eval

        return getattr(policy_eval, name)
    if name in _COMPARE_EXPORTS:
        from . import compare_policies

        return getattr(compare_policies, name)
    if name in _SEARCH_EVAL_EXPORTS:
        from . import search_eval

        return getattr(search_eval, name)
    if name in _BENCHMARK_SUITE_EXPORTS:
        from . import benchmark_suite

        return getattr(benchmark_suite, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
