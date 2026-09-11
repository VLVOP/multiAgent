# Retrieval Augmentation Design

## Goal

The retrieval layer serves biomedical Literature-Based Discovery (LBD), not generic question answering. Agents should retrieve biomedical entities, relations, and supporting literature under an explicit temporal cutoff.

The key principle is:

> Entity retrieval finds relation candidates; document retrieval finds relation evidence.

## Two retrieval layers

### 1. Entity-Relation Retrieval

Purpose: explore the ABC discovery space.

- Input: biomedical entity A or intermediate bridge entity B, cutoff year t, optional relation constraints.
- Output: candidate neighboring entities and relation candidates available from evidence no later than t.
- Main use: EXPLORE / bridge discovery / candidate generation.

Conceptual interface:

```text
resolve_entity(entity)
get_entity_info(entity)
expand_entity(entity, cutoff)
verify_relation(A, B, cutoff)
```

The corresponding representation is an Entity/Relation Graph:

- nodes = normalized biomedical entities;
- edges = literature-supported biomedical relations;
- edge metadata includes relation type, evidence PMIDs, earliest/supporting years, and confidence;
- relation edges must obey the temporal cutoff.

Ontology resources such as MeSH/UMLS identify and normalize entities; they do not by themselves establish whether a biomedical relation was known before the cutoff.

### 2. Document / Evidence Retrieval

Purpose: retrieve literature that supports, contradicts, or establishes prior knowledge of a candidate relation.

- Input: relation/query plus cutoff year.
- Output: PubMed/MEDLINE documents and evidence metadata.
- Main use: VERIFY / CRITIQUE / REFINE / novelty checking.

Conceptual interface:

```text
search_literature(query, cutoff)
get_evidence(A, B, cutoff)
search_counter_evidence(A, B, cutoff)
check_novelty(A, C, cutoff)
```

The corresponding representation is a Document/Evidence Graph or evidence store:

- documents are indexed by PMID and publication year;
- entity mentions/relations connect documents to the entity graph;
- supporting and contradictory evidence remain traceable to source documents.

The entity graph and evidence store are linked through normalized entity identifiers and PMIDs.

## Tool interface versus backend

Agents should depend on stable semantic tool interfaces rather than a specific database or API implementation.

```text
Agent
  -> Tool Interface
      -> Online Backend
      -> Frozen/Local Backend
```

Examples:

```text
search_literature(...)
resolve_entity(...)
expand_entity(...)
verify_relation(...)
check_novelty(...)
```

The Agent should not need to know whether the implementation is NCBI E-utilities, a local MEDLINE index, a local graph, or another compatible backend.

## Online backend

The online mode is intended for realistic agent operation and demos.

Current direction:

- PubMed/MEDLINE through NCBI E-utilities;
- MeSH for entity lookup/normalization;
- live entity-pair evidence retrieval;
- live novelty and counter-evidence queries.

The temporal cutoff must be enforced inside the tool/backend, for example:

```text
search_literature(query, before_year=1985)
```

The LLM must not be trusted to remove post-cutoff evidence after retrieval.

## Frozen / local backend

The paper's main benchmark experiments should use a reproducible frozen corpus/index rather than depend on live API results.

Desired properties:

- fixed PubMed/MEDLINE snapshot;
- deterministic preprocessing and indexing;
- publication-year filtering at query time;
- identical tool interfaces to the online backend;
- reproducible entity graph/evidence store construction;
- support for historical slices D_{<=t}.

This produces two modes:

```text
Online mode:     Tool Interface -> NCBI / external biomedical APIs
Benchmark mode:  Tool Interface -> frozen local snapshot/index
```

The same agents and state graph should run in both modes.

## Temporal grounding

For a cutoff year t:

```text
G_{<=t}
D_{<=t}
```

Only relation evidence and documents available no later than t may affect discovery, verification, novelty checking, or routing.

Entity normalization resources may be modern if used only to identify canonical entities, but relation/evidence knowledge must remain temporally grounded.

A useful rule is:

> Ontology tells us what the entity is; cutoff literature tells us what relation was known.

## Retrieval pipeline

The intended evidence pipeline is:

```text
retrieve candidate evidence
    -> learned ABC-state-conditioned evidence router
    -> Top-K supporting / contradictory / novelty-sensitive evidence
    -> LLM verification / critique
```

The learned router should eventually condition on:

```text
(A, B, C, discovery state S_t, current action a_t, document D_i)
```

rather than perform generic query-document reranking.

## Context, cache, and retrieval interaction

Retrieval should interact with the planned LBD Evidence Cache.

Example cache key:

```text
(A, B, cutoff_year)
```

Possible cached fields:

- supporting PMIDs;
- counter-evidence PMIDs;
- evidence/document embeddings;
- relation judgment;
- confidence;
- verification state;
- topic distribution / topic coverage metadata.

The sparse discovery policy can then choose among:

```text
retrieve / reuse / refine / explore
```

instead of repeating the same retrieval during every loop iteration.

LDA-based topic-aware context management can complement the learned evidence router: the router models task-specific relevance, while LDA contributes topic coverage, diversity, and redundancy control.

## Current implementation status

Implemented online first-pass tools include:

- PubMed literature search with cutoff;
- MeSH entity lookup;
- biomedical entity expansion via pre-cutoff literature/MeSH co-indexing;
- entity-pair evidence retrieval;
- pair-mention counting;
- entity information lookup;
- relation verification;
- counter-evidence search;
- novelty checking.

Still to implement for the paper-grade retrieval layer:

- unified backend abstraction / stable semantic tool interface;
- frozen/local PubMed/MEDLINE backend;
- reproducible entity/relation graph construction;
- stronger typed relation extraction/verification;
- learned ABC-state-conditioned evidence routing;
- evidence aggregation/scoring;
- cache-aware retrieval and context management.
