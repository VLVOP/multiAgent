# Router Generalization Setting

The Evidence Router should be treated as an **LBD task-level policy**, not as a dataset-specific classifier.

## Core assumption

The shared router is conditioned on the normalized LBD state and action:

```text
(A, B, C, S_t, a_t, D_i)
```

rather than on a dataset identifier.

The action space should be standardized across datasets, for example:

```text
verify_AB
verify_BC
check_AC_novelty
find_bridge
find_counter_evidence
```

The router predicts shared evidence roles such as:

```text
[s_support, s_counter, s_bridge, s_novelty]
```

This makes the router **task-conditioned / state-conditioned rather than dataset-conditioned**.

## Preferred evaluation protocol

Do not assume one separately trained router per benchmark. The preferred evaluation hierarchy is:

1. in-domain evaluation;
2. cross-dataset zero-shot transfer;
3. multi-dataset training;
4. leave-one-dataset-out evaluation;
5. optional few-shot calibration as a secondary setting.

A particularly important experiment is:

```text
Train: Dataset A + Dataset B + Dataset C
Test:  Dataset D
```

where the router has never seen Dataset D during training. Stable gains over BM25, dense retrieval, generic rerankers, and state-agnostic rerankers would support the claim that the model learns reusable LBD routing behavior rather than benchmark-specific shortcuts.

Another useful transfer test is:

```text
Train: Dataset A
Test:  Dataset B / Dataset C
```

and then compare it with multi-dataset training to study how router generalization changes as the diversity of LBD episodes increases.

## Architecture / model disentanglement

The paper should explicitly test:

```text
Dataset x Backbone LLM x Architecture Variant
```

The retrieval pool, frozen temporal corpus, tool interface, and evaluation protocol should be held fixed whenever possible. This is necessary to separate gains caused by the Sparse Agentic LBD architecture from gains caused by a stronger backbone LLM or a different retrieval environment.

A strong architecture-oriented comparison is:

```text
Small/medium LLM + Sparse Agentic LBD
vs.
Large LLM + vanilla agent / flat RAG
```

with both discovery quality and efficiency reported.

## Dataset adaptation policy

Dataset-specific adaptation should be optional and lightweight rather than the default. If distribution shift requires calibration, prefer a thin calibrator or adapter over retraining the whole router, for example:

```text
shared biomedical encoder
        -> shared ABC/state encoder
        -> shared action-conditioned router
        -> optional tiny dataset calibrator
```

This preserves a shared LBD routing policy while allowing small benchmark-specific score calibration if necessary.

## Important distinction

Different LBD task formulations may have different action distributions, for example open LBD versus closed LBD. This should be handled through the explicit action/state representation rather than by introducing a dataset-specific router.

The intended claim is therefore:

> The router is trained across LBD episodes and conditioned on discovery state and action, rather than specialized to individual datasets.
