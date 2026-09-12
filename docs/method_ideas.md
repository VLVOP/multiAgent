# Method Ideas

## Sparse Agentic LBD

Working method direction for the paper:

### 1. ABC-state-conditioned Evidence Router

Instead of generic query-document reranking, score candidate literature conditioned on the current biomedical LBD state:

- entities: A, B, C;
- current discovery state S_t;
- current action a_t, e.g. verify A-B, verify B-C, check A-C novelty, critique;
- candidate document D_i.

Conceptually:

```text
P(D_i | A, B, C, S_t, a_t)
```

The router can predict multiple evidence dimensions, for example support, contradiction, bridge relevance, and novelty sensitivity, then select Top-K evidence for expensive LLM reasoning.

The intended advantage is **lightweight and fast routing**: biomedical document embeddings can be precomputed and cached, while online inference only encodes the current discovery state and applies a small trainable routing head (for example a small attention/MLP module). Expensive LLM reasoning is therefore reserved for high-value evidence rather than every retrieved document.

### 2. Cache-aware Sparse Discovery Policy

Introduce an LBD-specific evidence cache keyed by temporally grounded entity relations, for example:

```text
(A, B, cutoff_year) -> {
  supporting_pmids,
  counter_pmids,
  evidence_embeddings,
  relation_judgment,
  confidence,
  verification_state
}
```

The discovery policy should decide whether to retrieve, reuse cached evidence, refine an existing relation, or explore a new branch.

A useful objective is to trade expected scientific discovery gain against computation cost:

```text
a_t = argmax_a ExpectedDiscoveryGain(a | S_t) - lambda * Cost(a)
```

This is especially important in iterative LBD loops, where VERIFY -> CRITIQUE -> REFINE -> VERIFY can otherwise repeat the same retrieval and verification work. Reflection can guide the policy by identifying which relation is strong, weak, contradictory, or unresolved, after which the policy reallocates computation instead of redoing already-solved work.

### 3. Infrastructure-inspired Sparse Computation

Borrow sparse-compute ideas from LLM infrastructure rather than treating all tools, agents, documents, and token budgets uniformly. The current ABC state may route:

- which evidence to inspect;
- which tool to invoke;
- which agent should act next;
- how much retrieval/LLM compute to allocate;
- whether cached evidence can be reused.

The intended contribution is not generic caching or generic routing by itself, but **ABC-aware routing + evidence caching + adaptive compute** inside iterative biomedical Literature-Based Discovery.

### 4. Topic-aware Context Management with LDA

Use Latent Dirichlet Allocation (LDA) as a lightweight context organizer rather than as the primary semantic retriever.

Each document, tool observation, reflection, or cached evidence item may carry a topic distribution:

```text
theta_d = P(z | d)
```

The current discovery state can maintain a corresponding topic profile:

```text
theta_S = P(z | S_t)
```

The context manager can then use topic information for:

- retaining context relevant to the current ABC state;
- maintaining topical diversity in the selected evidence;
- reducing redundant historical observations;
- deciding which older context can be compressed, cached, or discarded;
- organizing the evidence cache by topic coverage.

The learned Evidence Router should primarily model task-specific relevance, while LDA contributes topic coverage and diversity. A conceptual combined score is:

```text
Score(d) = alpha * RouterRelevance(d)
         + beta  * TopicCoverage(d)
         - gamma * Redundancy(d)
```

This can produce a **topic-aware evidence cache**, where cached entries store not only relation evidence but also topic distributions and topical coverage metadata.

LDA itself is not intended as a standalone contribution. The intended role is a lightweight context-management component inside **ABC-aware routing + evidence cache + adaptive compute**.

### 5. Structured Sparse A2A Communication

Do not treat A2A as a standalone protocol contribution. Instead, use **LBD-specific structured communication** between agents so that they exchange typed discovery state rather than long free-form histories.

A candidate message schema is:

```text
DiscoveryMessage {
  abc_path,
  relation_under_test,
  evidence_ids,
  support_score,
  contradiction_score,
  novelty_status,
  uncertainty,
  requested_action,
  provenance,
  cache_references
}
```

Examples:

```text
Explorer -> Verifier:
(B, C, evidence_ids, uncertainty, request=verify)

Verifier -> Critic:
(support, contradiction, novelty, confidence, provenance)
```

A lightweight message router may decide:

- which observations are worth communicating;
- whether to send raw evidence, a compact summary, or a cache reference;
- whether another agent should be triggered at all;
- which parts of the current ABC state are relevant to the receiving agent.

Conceptually:

```text
P(m_i | S_t, agent_src, agent_dst)
```

This turns communication into another sparse resource allocation problem. The intended framing is therefore not generic A2A, but **ABC-aware structured A2A + selective communication + evidence/cache references**.

This yields a unified sparse-system view:

```text
Sparse Evidence
+ Sparse Compute
+ Sparse Communication
```

Useful communication-efficiency metrics include:

- communication tokens;
- number of inter-agent messages;
- duplicated evidence transmitted;
- cache-reference hit rate;
- discovery quality under fixed communication budgets.

Natural baselines include full shared context, free-form A2A, structured A2A without routing, and structured sparse A2A.

### 6. Hierarchical Progressive Context Disclosure

The context design should not be restricted to a fixed two-layer scheme. Instead, organize available context into an **N-level hierarchy**, while treating progressive disclosure as a separate decision about what the current agent is allowed to see at a given step.

A possible hierarchy is:

```text
L0: current ABC discovery state / goal / uncertainty / prior conclusions
L1: entity-relation graph context, candidate paths, relation scores
L2: relation-level evidence summaries, support/counter counts, confidence
L3: document-level context, titles/abstracts/document cards
L4: fine-grained passages, experimental details, provenance/raw observations
...
LN: deeper evidence or source-specific context if required
```

The key distinction is:

```text
Select:     which information is worth retaining in the context store?
Disclosure: which retained information should this agent see now?
```

Thus the context store may already contain L0...LN, but an Explorer may only receive L0+L1, while a Verifier may start with L0+L1+L2 and request local L3/L4 evidence only when uncertainty, contradiction, novelty ambiguity, or an evidence gap appears.

Conceptually, a disclosure policy may be written as:

```text
pi_ctx(level, region | S_t, a_t, uncertainty_t)
```

Disclosure should be local rather than blindly expanding an entire layer. For example:

```text
Disclosure(L3, relation=B-C, docs={d1,d17,d31})
Disclosure(L4, document=d31)
```

This mechanism addresses the combinatorial context explosion inherent in LBD:

```text
Entity -> Relation -> Path -> Evidence -> Document -> Passage
```

The two retrieval stores naturally populate different levels of this hierarchy:

```text
Entity/Relation Retrieval -> coarse discovery and relation context
Document/Evidence Retrieval -> fine-grained evidence and document context
```

LDA supports topic coverage, diversity, and redundancy control within levels; the Evidence Router determines evidence relevance; progressive disclosure determines **when and how deeply** selected context should enter an agent's actual context window.

A GSSC-style context pipeline can be used as the implementation skeleton:

```text
Retrieval
  -> Gather
  -> Select
  -> Hierarchical Context Pool
  -> Progressive Disclosure Gate
  -> Structure
  -> Compress
  -> Agent
```

Compression must preserve evidence-critical content. Provenance, PMIDs, contradiction cues, negation, and critical evidence spans should not be aggressively generatively compressed.

Useful evaluation dimensions include context tokens, average disclosure depth, layer-access frequency, redundancy, topic coverage, and discovery quality under fixed context budgets.

### 7. Router Training and Temporal Agentic LBD Dataset

The first trainable component should be the Evidence Router rather than the entire agent policy. Its core task is:

```text
f_theta(S_t, a_t, D_i) -> evidence score(s)
```

A lightweight implementation can use a frozen biomedical document encoder with precomputed/cached document embeddings, plus a small trainable state encoder and fusion/routing head. This keeps online routing substantially cheaper than asking a large LLM to inspect every candidate paper.

#### Training instance

A training example should be **state-conditioned**, not merely a query-document relevance pair:

```text
A = Migraine
B = Vascular Tone
C = Magnesium
cutoff = 1981
current_action = verify(B, C)
candidate_document = PMID_xxx

labels:
  support
  contradiction
  bridge
  novelty_relevance
```

The central idea is that relevance changes with the current action. A paper that is useful for verify(A,B) may be low-value for verify(B,C), even inside the same ABC episode.

#### Multi-head routing target

A shared router may predict several evidence dimensions:

```text
[s_support, s_counter, s_bridge, s_novelty]
```

The current action determines how these heads are combined. For example:

```text
verify(B,C)      -> support + counter relevance
check_novelty    -> novelty relevance
explore_bridge   -> bridge relevance
```

This provides an action-conditioned multi-task evidence router rather than multiple unrelated rerankers.

#### Hard negatives

Random PubMed negatives are too easy. The dataset should include at least:

1. **Lexical hard negatives**: A/B/C terms appear but the target relation is unsupported.
2. **Path hard negatives**: evidence belongs to another edge of the same ABC path, e.g. A-B evidence while verifying B-C.
3. **Semantic hard negatives**: similar MeSH/entity types/topics but the relation is wrong.
4. **Temporal hard negatives**: scientifically relevant evidence that appears only after the cutoff and therefore must not be usable in the historical discovery state.

Hard negatives should be matched where possible on semantic type, frequency, graph degree, and temporal window so that the router cannot solve the task using trivial shortcuts.

#### Temporal episode format

Rather than storing only isolated query-document pairs, construct **Temporal Agentic LBD Episodes**:

```text
Episode
├── A
├── cutoff t
├── candidate B
├── candidate C
├── ABC state S_t
├── current action a_t
├── retrieved candidate pool
├── support evidence
├── counter evidence
├── hard negatives
└── post-cutoff future evidence
```

The same episode format can later support Evidence Router training, disclosure-policy learning, cache-policy learning, and agent-routing experiments.

#### Initial training objective

A practical supervised starting point is:

```text
L = L_rank + lambda_1 * L_type + lambda_2 * L_temporal
```

where:

- L_rank ranks useful evidence above hard negatives;
- L_type supervises support/counter/bridge/novelty roles;
- L_temporal penalizes use of post-cutoff evidence in historical states.

RL is not required initially. A supervised router should first establish strong Top-K evidence quality and compute savings; more global discovery-gain/cost optimization can be explored later.

#### Router evaluation

Important baselines include:

```text
BM25 Top-K
Dense Top-K
Generic biomedical reranker
State-agnostic learned reranker
ABC-state-conditioned Router
```

Report both evidence/discovery quality and efficiency:

```text
Recall@K / MRR / nDCG / Evidence Precision
LLM tokens
number of documents passed to the LLM
latency
router inference cost
```

The intended result is a better quality-cost Pareto frontier: similar or better discovery quality while substantially reducing expensive evidence inspection.

### Current framing

A possible umbrella term is:

**Sparse Agentic LBD**

with three primary method contributions:

1. **ABC-State-Conditioned Evidence Routing**: use a lightweight, action-conditioned router to select evidence worth expensive reasoning and address evidence overload.
2. **Reflection-Guided Cache-Aware Sparse Discovery Policy**: reduce iterative redundant computation by deciding what to retrieve, reuse, refine, or explore based on the evolving discovery state.
3. **Demand-Driven Hierarchical Progressive Context Disclosure**: address combinatorial context explosion by organizing context into multiple levels and exposing deeper/local evidence only when the current agent needs it.

Topic-aware LDA management and structured sparse A2A strengthen the same sparse-discovery story rather than being treated as disconnected standalone claims.

The three corresponding problem statements are:

```text
Evidence Overload                  -> Lightweight ABC-aware Routing
Iterative Redundant Computation    -> Cache-aware Sparse Discovery Policy
Combinatorial Context Explosion    -> Hierarchical Progressive Disclosure
```

These ideas should be evaluated against full-retrieval/all-agent baselines, fixed Top-K, generic rerankers, no-cache, LRU/semantic-cache baselines, generic learned routers, communication-routing baselines, and flat/full-context prompting, while reporting discovery quality together with compute/tool/token/communication/context cost.
