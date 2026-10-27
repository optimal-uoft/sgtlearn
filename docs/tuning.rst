Hyperparameter tuning
=====================

Start with the estimator defaults
-------------------------------------

``SGTClassifier()`` and ``SGTRegressor()`` already use the shared regularized
settings selected from the study. Ordinary fitting runs ShapeCART with TAO
and bivariate branching disabled. No parameter helper is needed for a default fit.

.. list-table:: Shared tree defaults (also used by forest base trees)
   :header-rows: 1
   :widths: 65 35

   * - Parameter
     - Default
   * - ``max_depth`` / ``max_leaf_nodes``
     - ``4`` / ``None``
   * - ``num_partitions``
     - ``7``
   * - ``inner_max_depth`` / ``inner_max_leaf_nodes``
     - ``None`` / ``4``
   * - ``min_samples_leaf`` / ``inner_min_samples_leaf``
     - ``4`` / ``13``
   * - ``min_impurity_decrease`` / ``inner_min_impurity_decrease``
     - ``1.0`` / ``0.01``
   * - ``coordinate_descent_max_iters`` / ``coordinate_descent_patience``
     - ``5`` / ``5``
   * - ``pairwise_candidates`` / ``pairwise_penalty`` / ``branching_penalty``
     - ``0`` / ``0.0`` / ``0.0``
   * - ``tao_n_runs`` / ``tao_lambda`` / ``tao_pair_scale``
     - ``0`` / ``1e-4`` / ``2.0``

Classifier criterion is ``gini``; regressor criterion is ``squared_error``.
Use ``pairwise_candidates=0.5`` as the study-recommended starting point when
enabling bivariate branching directly. Very small datasets may need lower
minimum leaf sizes or impurity penalties to grow beyond a constant tree.
The study's reported scores used both bivariate branching and TAO; they are
not measurements of the final default preset.

Tune with Optuna
----------------

Install the optional Optuna dependency:

.. code-block:: console

   pip install 'sgtlearn[tune]'

The :doc:`tutorials/hyperparameter-tuning` notebook runs small classification
and regression searches, reconstructs the winning dictionaries, and evaluates
on held-out test data. It includes executed outputs and needs no data download.

Pass an Optuna trial to :func:`sgtlearn.tune.suggest_params` to obtain a
dictionary for either single-tree estimator. The helper only suggests parameters;
you control data splitting, scoring, the sampler, and the optimization budget.

.. code-block:: python

   import optuna
   from sklearn.model_selection import cross_val_score
   from sgtlearn import SGTClassifier
   from sgtlearn.tune import suggest_params

   def objective(trial):
       params = suggest_params(trial)
       model = SGTClassifier(**params, random_state=42)
       return cross_val_score(
           model, X_train, y_train, cv=3, scoring="balanced_accuracy"
       ).mean()

   study = optuna.create_study(direction="maximize")
   study.optimize(objective, n_trials=100)
   params = suggest_params(optuna.trial.FixedTrial(study.best_trial.params))
   model = SGTClassifier(**params, random_state=42).fit(X_train, y_train)

Keep final test data outside the objective. For regression, use
``SGTRegressor`` and an appropriate regression score such as
``neg_mean_squared_error``. The returned dictionary includes fixed parameters,
so reconstruct it with ``FixedTrial`` instead of using ``study.best_params``
alone. Override entries in the dictionary when exploring outside the suggested
ranges. Estimator parameters outside the helper remain available directly.

``suggest_params`` accepts ``use_tao`` and ``use_bivariate`` flags, both
defaulting to ``False``. Either can be enabled independently.
With bivariate branching enabled it searches positive pair budgets and pair
penalties; otherwise neither is sampled. TAO uses the fixed recommended
settings (one sweep, lambda ``1e-4``, pair multiplier ``2.0``) when enabled.
Reconstruct a winning trial with the same flags, and
use a separate study when changing the flags:

.. code-block:: python

   params = suggest_params(trial, use_tao=True, use_bivariate=True)
   # After optimizing with those same flags:
   params = suggest_params(
       optuna.trial.FixedTrial(study.best_trial.params),
       use_tao=True,
       use_bivariate=True,
   )

The flags belong to ``suggest_params``, not the estimator constructors. Pass
the resulting dictionary with ``SGTClassifier(**params)`` or
``SGTRegressor(**params)``. Keep the same flags throughout each study and
when reconstructing its winning trial. TAO controls remain fixed rather than
being searched. ``use_bivariate=True`` searches budgets ``[.25, .5, 1, 2]``;
it does not fix the budget at ``0.5``.

For forests, pass the same dictionary to ``RandomSGForestClassifier`` or
``RandomSGForestRegressor``. It supplies tree parameters and preserves the
forest's feature-sampling defaults. Tree tuning does not establish optimal
forest settings; validate the final forest separately on your own data.

Regression target units
-----------------------

``SGTRegressor.fit`` standardizes each target column using its training mean
and population variance, respecting ``sample_weight``. Constant columns use a
safe unit scale. Each forest tree fits its own transformation on its bootstrap
sample. Predictions and exported leaf values use original target units;
impurity values and all regression growth and TAO penalties use standardized
units. Multi-output losses average over standardized outputs, preventing a
large-unit output from dominating the others.

Supply original-unit targets to later ``TAO_refine`` calls. Refinement reuses
the training transformation, including when refining a forest. A successful
refit replaces that transformation; a failed refit preserves the previous model.
The supported finite float32 input range remains unchanged.

Default-selection study
-----------------------

The study-selected shared configuration is outer depth 4, at most 7 branches,
4 inner leaves with unlimited inner depth, outer/inner minimum leaf sizes
4/13, outer/inner impurity penalties 1.0/0.01, and a pair-candidate budget of
``0.5 * n_logical_features`` (rounded up). Coordinate descent uses 5 iterations
and patience 5. TAO uses one sweep, ``tao_lambda=1e-4``, and pair multiplier 2.0.
Pair and extra-branch penalties are zero. Classifier criterion stays ``gini``;
regressor criterion stays ``squared_error``.

Public constructors keep bivariate splits and TAO off by default
(``pairwise_candidates=0``, ``tao_n_runs=0``). Set ``pairwise_candidates=0.5``
to opt into the study-recommended bivariate starting point. The other settings
match the selected configuration. The reported study scores used bivariate
splits and one TAO sweep; they do not measure the final univariate,
ShapeCART-only defaults. The tuning helper leaves TAO and bivariate branching
disabled unless their respective flags are enabled. Refinement can also be applied
explicitly with ``TAO_refine``.

Forest base trees inherit these settings. Existing forest-specific defaults
match sklearn: 100 trees, bootstrap enabled, all rows per bootstrap sample,
``max_features="sqrt"`` for classification and ``1.0`` for regression,
``random_state=None``, ``n_jobs=None``, and ``verbose=0``. Forest performance
was not measured in this study.

This changes previous constructor behavior through the regularized tree settings.
Very small datasets can produce a constant tree under the regularized defaults;
lower the outer/inner minimum leaf sizes and impurity penalties when needed.

The study scripts in ``benchmarks/tune_defaults.py`` and
``benchmarks/analyze_tuning.py`` record every dataset and fold for every trial.
Twelve TabArena datasets are used for optimization, with six classification
and six regression tasks ranging from 748 to 13,776 rows. Three inner folds are
created within each official repeat-0/fold-0 training set; the corresponding
official test set is reserved for confirmation. Two additional datasets are
excluded from optimization altogether. This is a small custom study using
TabArena datasets, not a full TabArena leaderboard evaluation.

The expanded panel adds blood transfusion, website phishing, churn, QSAR fish
toxicity, healthcare insurance expenses, used Fiat 500 prices, and Miami
housing to the original credit-g, diabetes, maternal health risk, airfoil,
and concrete datasets. QSAR biodegradation and wine quality remain excluded
from optimization. The original five-dataset study is retained separately;
its Pareto configurations are reevaluated on the expanded panel as warm starts.

The two minimized objectives are equally weighted mean classification error
(``1 - balanced_accuracy``) across classification datasets and mean regression
``MSE / training_target_variance`` across regression datasets. One shared
configuration is selected from the Pareto front by normalizing each objective's
range on that front and minimizing Euclidean distance to the ideal point.
Ties are resolved by trial number. Runtime is reported, not optimized.

TAO stays enabled, ``tao_lambda=1e-4``, ``inner_max_depth=None``, and outer depth
is searched from 4 through 10. The initial inner-leaf search spans 2 through
256. Parameter importance estimates are supplemented by one-parameter
validation sweeps around the selected configuration. These local sweeps suggest
useful ranges, but do not establish universal causal importance or cover every
interaction. Forests are not benchmarked; they inherit tree settings by design.
Each trial must finish every search dataset within a ten-minute configuration
limit. Timed-out trials receive no objective values and are reported as failures.

Useful parameters to tune
-------------------------

The public helper explores outer depth 4--10, inner leaves 2--64 (log scale),
outer branches 2--8, and outer/inner minimum leaf sizes 1--32 (log scale).
It also samples outer impurity penalties from ``[0, .01, .1, 1, 10]``, inner
penalties from ``[0, .001, .01, .1]``, pair-candidate budgets from
``[.25, .5, 1, 2]`` when enabled, and pair/extra-branch costs from
``[0, .01, .1, 1]``.
Inactive costs are not sampled for univariate or binary-only settings.

Outer impurity regularization and minimum leaf sizes were the strongest
classification dimensions. Inner minimum leaf size, outer impurity cost, and
inner leaf count were the strongest regression dimensions. Iteration count,
patience, TAO sweep count, and TAO pair multiplier showed little local benefit
around the selected configuration. The helper fixes these controls, with TAO
disabled unless requested with ``use_tao=True``.
The original broad search covered inner leaves up to 256; the local sweeps
showed plateaus at larger values, so the public helper uses 64 as its upper
bound. Users can still override the returned dictionary.

The expanded study completed 512 trials (325 distinct parameter dictionaries).
Its reserved classification tests were mixed: mean balanced accuracy fell
from 0.7306 to 0.7199, while mean normalized regression MSE improved from
0.3932 to 0.2592. On the two datasets excluded from optimization, QSAR
biodegradation balanced accuracy improved from 0.6783 to 0.7860, and wine
quality MSE improved from 0.7969 to 0.5140. These are single-split estimates,
not a guarantee for every new dataset. Full per-dataset results, recommended
tested values, and the Pareto plot are recorded in the repository's
``docs/research/2026-10-26-tuning-defaults.md`` report.

.. autofunction:: sgtlearn.tune.suggest_params
