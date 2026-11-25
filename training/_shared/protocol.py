"""Experiment settings shared by the command line, trainer and verifier."""

TEST_YEARS = tuple(range(2015, 2020))
TUNING_YEAR = TEST_YEARS[0]
TUNING_VALIDATION_YEARS = [TUNING_YEAR - 2, TUNING_YEAR - 1]
TUNING_SEED = 42
SEEDS = (42, 123, 2024, 3407, 7890)
SEARCH_RATES = (0.0003, 0.001, 0.003)
PROTOCOL = 'fixed_initial_lr_then_fold_early_stop'

JOINT_ABLATIONS = ('self_loop_no_sif', 'dynamic_only')
SELF_LOOP_VARIANTS = ('self_loop', *JOINT_ABLATIONS)
NEURAL_BASELINES = ('single', 'lstm_daily', 'transformer_daily', 'stgcn')
NON_NEURAL = ('ema_only', 'rf')
NO_GRAPH_VARIANTS = ('no_graph', 'no_graph_no_sif')
MATCHED_TRANSFORMERS = ('transformer_matched', 'transformer_matched_no_context')
VARIANTS = [
    'full', 'transformer_matched', 'transformer_dynamic', 'transformer_original9',
    'self_loop', 'no_sif', 'sif_concat', 'ema_only', 'no_ema', 'ema_03', 'ema_07',
    'no_static', 'flat_temporal', 'static_concat', 'no_residual', 'fixed_residual',
    'graph_distance', 'graph_environment', 'graph_random', 'stage_early', 'stage_late',
    *JOINT_ABLATIONS, *NO_GRAPH_VARIANTS, 'transformer_matched_no_context', *NEURAL_BASELINES, 'rf',
]
