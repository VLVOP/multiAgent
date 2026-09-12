# Adaptive Hierarchical Context for Agentic LBD

## Positioning

The context subsystem may deliberately use an L1/L2/L3-style global memory hierarchy. The hierarchy itself is useful system infrastructure; the methodological contribution is the **LBD-specific organization plus state-conditioned, agent-specific access/disclosure policy**.

A concrete hierarchy is:

```text
L1: Global discovery state
    ABC path, current goal, loop state, uncertainty, prior decisions

L2: Structured relation memory
    entity relations, bridge candidates, evidence summaries, confidence,
    cache status, support/counter counts

L3: Detailed evidence memory
    PMIDs, titles/abstracts, evidence passages, experimental details,
    provenance and raw tool observations
```

More levels can be introduced if evaluation shows a need, but the design should not depend on a fixed number of levels.

## Agent-specific views

Different agents should receive different default views:

```text
Planner      -> L1
Explorer     -> L1 + selected L2
Hypothesis   -> L1 + selected L2
Verifier     -> L1 + L2 + on-demand L3
Critic       -> L1 + L2 + disputed/contradictory L3
```

The important distinction is between information that exists in global context memory and information actually disclosed into one agent's context window.

## Adaptive disclosure

Access is conditioned on the current LBD state and action rather than on a static prompt template. Typical triggers for deeper disclosure include:

- unresolved A-B or B-C support;
- A-C novelty ambiguity;
- contradictory evidence;
- high uncertainty;
- a reflection/refinement request for a specific relation.

Thus the methodological object is approximately:

```text
pi_ctx(level, region | S_t, agent, action, uncertainty)
```

The policy should support local disclosure, e.g. expanding only the disputed B-C relation or one contradictory PMID rather than all documents at the same depth.

## Relationship to other components

- Evidence Router decides which retrieved evidence is valuable.
- Hierarchical Context Manager decides where that evidence is represented and what each agent sees.
- Progressive Disclosure decides when a deeper level becomes visible.
- Cache avoids recomputing already established relation evidence.
- Structured A2A should pass references/typed state rather than duplicate full context.

The intended paper framing is therefore **Adaptive Hierarchical Context for Agentic LBD**, not merely generic hierarchical memory.

## Evaluation

Compare at least:

```text
Full shared context
Fixed truncation
Static hierarchy
Adaptive hierarchy/disclosure
```

Report discovery quality together with context tokens, disclosure depth, duplicated evidence, layer-access frequency, and cost.
