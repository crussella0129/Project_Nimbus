# ENGRAMNET — Specification and Build Intents (v0.1)

*Working name `engramnet` (verify crates.io availability before publishing; fallback prefix `egn-`). This document is both a design spec and a set of executable intents for Claude Code. It is derived from the research in `report_budgeted_networks.md` and `report_addendum_engram_runner.md` (same folder); the numeric "golden" values below come from `experiments/`.*

---

## 0. What we are building, in one paragraph

A Rust-first, open-source framework for **budgeted any-to-any neural networks** — graphs of bounded units (activations in [−1, 1]) in which every node distributes a fixed *influence budget* over its outgoing connections (Σⱼ w²ᵢⱼ ≤ g², all |w| ≤ 1), any node may connect to any node including itself, and a **runner** selects which nodes to drive at run time from a **context-keyed engram table** (hashed n-gram keys → sparse node cues, gated by similarity to the current state) so the network completes ("intuits") the stored association. Around that core: spectral/stability tooling (because the budget does *not* buy stability by itself), three training modes (fixed / learned / global budget, plus learnable topology), associative-memory layers, a **weight-space programming** interface (legible export, bounded and audited edits, LLM-agent harness), and an optional adapter that applies the same conditional, gated node-selection to real transformers' FFN neurons (replacing hand-picked, unconditional "levers").

## 1. Design principles (each traces to a research finding)

| # | Principle | Why (evidence) |
|---|---|---|
| P1 | **Bounded by construction.** Activations and weights are bounded types; budget invariants are enforced by projection after every mutation and checked in tests. | Max-norm/weight-norm/nGPT/Lipschitz-transformer line; our Exp. 4 shows no accuracy cost at toy scale. |
| P2 | **Separate *which* from *how much*.** The runner/table decides which nodes are driven; the budget bounds how much any driven node can do. | Lyra-style levers entangle both in one dose and loop at dose ≥1.5; CAST/Engram separate condition from action. |
| P3 | **Stability is measured, never assumed.** Ship ρ(W), σ_max(W), Lyapunov estimators and regime classification as first-class APIs; a unit fan-out budget sits exactly at the edge of chaos (ρ ≈ 1, σ_max ≈ 2) and structured graphs can reach σ_max = √N. | Sompolinsky et al. 1988; Schoenholz et al. 2017; Yildiz et al. 2012; our Exp. 1–2. |
| P4 | **Interventions are conditional and gated by default.** No unconditional steering API in the public surface; every injection passes a gate α ∈ [0,1] computed from context similarity. | CAST (ICLR 2025); DeepSeek Engram gate; alignment concerns raised by the user. |
| P5 | **Legible weight space.** Every weight is a share (w² = fraction of a node's budget); export is human/LLM-readable; edits are patches with a provable blast-radius bound (‖Δw_i‖·|x_i| ≤ 2). | Weight-space programming (Takehi et al. 2026); nGPT legibility of unit-norm vectors. |
| P6 | **Memory is collective.** Single self-loops cannot latch under |w| ≤ 1; provide motif primitives (hub + k≥2 spokes) and capacity tests (classical ~0.14·N limit; modern-Hopfield retrieval for exponential capacity). | Our Exp. 3 and Exp. 5; Ramsauer et al. 2021. |
| P7 | **Rust first; Python only at the edges** (bindings, HF-transformers adapter, notebooks). Core crates have no Python or libtorch dependency. | User requirement; ecosystem state (Burn 0.22 training; faer; safetensors/tokenizers crates). |
| P8 | **Reproduce known results as tests.** Golden tests encode the edge-of-chaos transition, the latch motif, Hopfield capacity, and Glorot's backward condition. | Keeps the framework honest as it grows. |

---

## 2. Recommended stack

### 2.1 Languages and toolchain
- **Rust stable** (edition 2024), MSRV pinned in `rust-toolchain.toml`; `cargo-nextest`, `cargo-deny`, `cargo-semver-checks`, `cargo-release`, `cargo-fuzz` (nightly only in CI fuzz job).
- **Python ≥ 3.10** only for: `engramnet` wheels (pyo3 + maturin), the HF-transformers adapter, example notebooks.

### 2.2 Crates by role (and why)

| Role | Crate | Rationale / notes |
|---|---|---|
| Dense tensors (CPU core) | `ndarray` (+ `rayon` feature) | Lightweight, no GPU dependency in the core; f32 default, f64 for spectral analysis. |
| Linear algebra: eigenvalues, singular values | `faer` | Pure Rust, parallel, non-Hermitian eigendecomposition and SVD (needed for ρ and σ_max). |
| Sparse matrices (N ≥ 10⁵ nodes) | `sprs` (CSR/CSC) | For sparse topologies from RigL-style rewiring. |
| Autodiff / training / GPU | `burn` 0.22 (`wgpu`, `cuda`, `metal` features; `flex` for CPU/no-GPU tests) + `burn-train` | Runtime device selection, autodiff via `device.autodiff()`, CubeCL backends, SafeTensors via `burn-store`. tch/ndarray backends are deprecated — do not use them. |
| HF LLM inference in Rust (adapter v2) | `candle-core`/`candle-nn`/`candle-transformers` (+ community `crane` for Qwen3.5) | Candle is inference-focused and HF-native; Qwen3.5 is *not* in upstream candle (open issue) but Crane supports it. Adapter v1 is Python (see §6, Phase 8). |
| Model files | `safetensors` (official HF crate), `memmap2` | Zero-copy, no pickle. All tables and graphs ship as safetensors + JSON sidecar. |
| Tokenization | `tokenizers` (HF crate) | For n-gram keys over real tokenizers; implement Engram-style canonicalisation (NFKC/lowercase) on top. |
| Hashing | `xxhash-rust` (xxh3) + in-crate multiplicative-XOR hash | Engram uses multiplicative-XOR with K heads over prime-sized tables; we implement that exactly, and use xxh3 for metadata. |
| ANN / product-key lookup (optional) | `usearch` or `hnsw_rs` | For learned-key memories (Phase 5b). Feature-gated. |
| Graph utilities | `petgraph` | Topology analysis (DAG detection ⇒ ρ = 0; cycle enumeration for latch motifs). |
| RNG | `rand` + `rand_chacha` | Seeded, reproducible across platforms. |
| Serialization/config | `serde`, `serde_json`, `rmp-serde` (MessagePack for traces), `toml` | |
| CLI | `clap` (derive) | `engramnet` binary: build / analyze / run / patch / export. |
| Logging/metrics | `tracing`, `tracing-subscriber`, `metrics` | Structured traces of runner steps (key, candidates, gates, injections). |
| Tests | built-in `#[test]`, `proptest` (invariants), `insta` (snapshot of readable exports), `approx` (float tolerances), `criterion` (benches), `cargo-fuzz` (parsers) | |
| Python bindings | `pyo3`, `maturin`, `numpy` (crate) | Expose `Network`, `Runner`, `EngramTable`, `Patch`. |
| Docs | `mdbook` (the "Engramnet Book"), docs.rs | Book chapters mirror the phases below. |
| HTTP (agent harness) | `reqwest` + `tokio` (feature `agent`) | Calls any OpenAI-compatible endpoint for weight-space-programming experiments; **off by default**. |

**Why not tch-rs / PyTorch bindings?** They drag libtorch into every build and Burn has deprecated that path; the core must stay pure Rust.
**Why Burn over Candle for training?** Burn has the training loop, autodiff-as-device-state, and multi-backend story; Candle is better for loading HF checkpoints, which only the adapter needs.

### 2.3 Workspace layout

```
engramnet/
  Cargo.toml                     # workspace, shared [workspace.dependencies], lints
  rust-toolchain.toml
  crates/
    engramnet-core/              # types, invariants, dense/sparse weights, dynamics, injection, safetensors I/O (no burn)
    engramnet-spectral/          # rho, sigma_max, Lyapunov, regime classifier, calibrate_gain (faer)
    engramnet-motifs/            # latch(k), chain, ring, hub; DAG checks (petgraph)
    engramnet-memory/            # Hebbian store, cue recall, modern-Hopfield retrieval, capacity sweeps
    engramnet-engram/            # tokenizer canonicalisation, multi-head hashed n-gram table, gating, cache tiers
    engramnet-runner/            # the decoder/runner loop, policies, traces
    engramnet-train/             # burn modules: budget projection (A), fan-out weight-norm (B), global budget (C), RigL rewiring
    engramnet-wsp/               # readable export, Patch/AuditRecord, blast-radius guard, optional `agent` feature
    engramnet-cli/               # `engramnet` binary
    engramnet-py/                # pyo3 bindings -> Python package `engramnet`
  adapters/
    hf-transformers/             # Python: gated FFN-neuron injection for HF models (reference implementation)
  book/                          # mdBook
  benches/                       # criterion
  examples/
  experiments/                   # the NumPy scripts from the research (kept as oracles for golden tests)
  .github/workflows/             # ci.yml (fmt, clippy -D warnings, nextest, doc, deny), bench.yml, release.yml, wheels.yml
```

Dependency direction (strict, enforced by `cargo-deny` bans and review): `core ← spectral ← motifs ← memory ← engram ← runner ← wsp ← cli/py`; `train` depends on `core` and `burn` only. **`engramnet-core` must compile with no default features in < 10 s and must not depend on burn, candle, or tokio.**

---

## 3. Core data model (engramnet-core)

```rust
/// Node index. u32 keeps dense matrices and tables compact.
pub struct NodeId(pub u32);

/// Budget constraint. "Fixed" variants are enforced by projection; Global is enforced by scaling.
pub enum Budget {
    FanOut { g: f32 },          // Σ_j w_ij² ≤ g² for every source node i  (the user's proposal; ChatGPT's A)
    FanIn  { g: f32 },          // Σ_i w_ij² ≤ g² for every target node j   (max-norm / weight-norm convention)
    Learned { g: Vec<f32>, axis: Axis }, // per-node learned g (ChatGPT's B); trained in engramnet-train
    Global { total: f32 },      // ‖W‖_F ≤ total (ChatGPT's C)
    Unconstrained,
}

/// Dense (row-major, W[i][j] = weight from i to j) or CSR storage. Both implement `Weights`.
pub trait Weights {
    fn n(&self) -> usize;
    fn matvec_t(&self, x: &[f32], out: &mut [f32]);   // out = Wᵀ x  (incoming sums)
    fn out_norms(&self) -> Vec<f32>;
    fn in_norms(&self) -> Vec<f32>;
    fn project(&mut self, budget: &Budget);             // enforce invariants (also clamps to [-1, 1])
    fn check(&self, budget: &Budget, tol: f32) -> Result<(), Violation>;
}

pub enum Activation { Tanh, HardTanh, Softsign }      // all map R -> [-1, 1]
pub enum Update { Synchronous, Leaky { alpha: f32 } } // x <- (1-a)x + a·φ(Wᵀx + u)

pub enum InjectMode { Clamp, Add }
pub struct Injection { pub node: NodeId, pub value: f32, pub mode: InjectMode, pub hold: u32 /* steps */ }

pub struct Network { pub w: Box<dyn Weights>, pub act: Activation, pub update: Update, pub budget: Budget }
impl Network {
    pub fn step(&self, x: &mut [f32], inputs: &[Injection]);
    pub fn relax(&self, x: &mut [f32], steps: usize, inputs: &[Injection]) -> RelaxReport; // energy-free convergence stats
}
```

Invariants (must hold after **any** public mutation; property-tested):
1. `|w_ij| ≤ 1` always (hard clamp after projection).
2. Under `FanOut{g}`: `out_norms()[i] ≤ g + 1e-5` for all i; equality for nodes with ≥1 edge when `exact=true`.
3. Under `FanIn{g}`: mirror of (2) on `in_norms()`.
4. Under `Global{total}`: `‖W‖_F ≤ total + 1e-5`.
5. State always in [−1, 1]ⁿ after `step`.
6. Diagonal (self-loops) allowed unless `Network.no_self_loops == true`; then `w_ii = 0` is an invariant.

File format: `graph.safetensors` with tensors `W` (dense) or `W.indptr/W.indices/W.data` (CSR), `budget.g` (per-node when learned), plus `graph.json` sidecar `{n, activation, update, budget, no_self_loops, created_by, version}`. The safetensors header is the only metadata channel permitted for strings (`__metadata__` is string→string), so structured config goes in the sidecar.

---

## 4. Runner architecture (engramnet-engram + engramnet-runner)

```
tokens (u32) ──► canonicalise ──► suffix n-grams (n=2..N) ──► K hash heads ──► table rows
                                                                  │
state x (N) ──► state key s = normalize(P·x)  ◄── gate α = σ((s·k̂)/τ) or cosine-threshold
                                                                  │
                                  candidates (cue = sparse {node: value}) × α  ──► Injection list
                                                                  │
                                            Network.relax(x, steps, injections) ──► readout
```

- **Key side (Engram-style):** canonical token IDs (NFKC + lowercase + merge table), suffix n-grams of orders 2..N, per order K hash heads `φ_{n,k}` (multiplicative-XOR, prime-sized tables), concatenate retrieved rows. Deterministic: the addresses depend only on tokens, so a *prefetch* stage can run ahead of the network.
- **Gate (CAST/Engram-style):** each table row stores a unit-norm `key_vec` (what context it is for) and a `cue` (which nodes to drive). α = sigmoid(⟨RMSNorm(s), RMSNorm(key_vec)⟩/√d) or a thresholded cosine; mismatched rows get α → 0.
- **Inject:** survivors become `Injection{Clamp|Add, hold}` scaled by α and by a global `dose ≤ 1`; the budget then bounds downstream effect (‖Δ‖ ≤ 2 per driven node per step).
- **Relax:** `steps` leaky updates; `RelaxReport` records convergence, oscillation, or divergence flags (from engramnet-spectral's regime classifier run on the local Jacobian).
- **Trace:** every step emits `{token, keys, candidates, alphas, injections, relax_report}` (MessagePack) — this is the audit log and the dataset for later learning of gates.

Capacity note: cue-recall in a budgeted Hebbian graph is reliable up to P ≈ 0.05·N stored patterns with a 10 % cue and collapses near 0.14·N; for larger stores use `engramnet-memory::Hopfield` (softmax retrieval, exponential capacity) to *select* the cue, then the graph to *act*.

---

## 5. Phases, recipes, tests (Claude Code intents)

Conventions for every phase: one PR per phase (sub-PRs allowed); tests written **before** implementation where feasible; `cargo fmt`, `cargo clippy --all-targets -D warnings`, `cargo nextest run --workspace`, `cargo doc --no-deps -D warnings`, `cargo deny check` must pass; add a chapter to `book/`. **Definition of done** = all listed tests pass in CI on Linux + macOS, benchmarks recorded, CHANGELOG entry added.

### Phase 0 — Bootstrap
**Intent:** a publishable, CI-green workspace skeleton.
- Create workspace, crates (empty libs with `#![deny(missing_docs)]`), `rust-toolchain.toml`, workspace lints (`unsafe_code = "forbid"` except in `engramnet-py`), dual license Apache-2.0 OR MIT, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, `docs/adr/0001-rust-first.md`, `0002-budget-semantics.md`, `0003-gating-by-default.md`.
- CI: fmt/clippy/nextest/doc/deny matrix (ubuntu, macos; stable + MSRV); fuzz job (nightly, allowed to fail initially); release-drafter.
- Copy `experiments/*.py` into repo as oracles; add `scripts/regenerate_golden.py` producing `tests/golden/*.json`.
**Tests:** CI passes on an empty workspace; `cargo deny` bans `tch`, `torch-sys`, `pyo3` outside `engramnet-py`.

### Phase 1 — Core types, invariants, I/O (engramnet-core)
**Intent:** bounded weights and budgets that cannot be violated through the public API.
- Implement `Budget`, `Weights` (Dense + CSR), `project`, `check`, `Activation`, `Update`, `Injection`, `Network::step/relax`, safetensors I/O (+ sidecar), seeded random initialisers (`unit_outgoing(g)`, `unit_incoming(g)`, `glorot_like`).
- `project` for `FanOut`: divide each row by max(1, ‖row‖/g) (inequality) or normalise exactly (`exact`); then clamp to [−1, 1] and re-check.
**Tests:**
- proptest: for random matrices and random sequences of ops (`set`, `add_edge`, `remove_edge`, `scale`), invariants 1–6 hold after `project`.
- `unit_outgoing(1.0)` ⇒ all out-norms = 1 ± 1e-6; Glorot equivalence: expected per-entry variance ≈ 1/fan_out.
- safetensors round-trip is bit-exact; sidecar round-trip; malformed files return `Err` (fuzz target `fuzz_load_graph`).
- `step` keeps state in [−1, 1]; `Clamp` injection pins values; `hold` releases after k steps.
- Bench: dense N=4096 step < 5 ms (rayon, 8 cores); CSR N=10⁶, avg degree 64, step < 200 ms.

### Phase 2 — Spectral & dynamics tooling (engramnet-spectral, engramnet-motifs)
**Intent:** make ChatGPT's "spectral properties matter" caveat a measured quantity.
- `spectral(&Weights) -> {rho, sigma_max}` via faer (f64); `lyapunov(&Network, steps, seed)` (Benettin re-normalisation); `regime(&Network) -> Quiescent|Oscillatory|Chaotic|Divergent` (activity + λ + recurrence); `calibrate_gain(&mut Network, target_rho)`; `is_dag` (petgraph) ⇒ ρ = 0 shortcut; motifs: `chain(n)`, `ring(n)`, `hub_spokes(k)`, `sink(n)`.
**Golden tests (tolerances from `experiments/exp1_results.json`, `exp2_results.json`):**
- Random `FanOut{g}` nets, N=800, 5 seeds: ρ ≈ g (±0.05), σ_max ≈ 2g (±0.05) for g ∈ {0.5, 1.0, 1.5}.
- `sink(100)`: σ_max = 10.0 ± 1e-6, ρ = 1.0 ± 1e-6. `hub_spokes(k)`: ρ = k^{1/4} ± 1e-6 for k ∈ {1,2,4,16,64}. `chain(100)`: ρ < 1e-9.
- Synchronous N=400: g=0.6 ⇒ activity < 1e-6 and λ = ln(0.6) ± 0.05; g=1.0 ⇒ activity 0.15 ± 0.05 and |λ| < 0.01; g=1.5 ⇒ λ = 0.054 ± 0.02; g=2.0 ⇒ λ = 0.149 ± 0.03; λ monotone non-decreasing in g.
- Leaky (α=0.3) `hub_spokes(k)` from hub=0.05: k=1 ⇒ |x_hub| < 0.1 after 20 000 steps; k=2 ⇒ 0.749 ± 0.01; k=4 ⇒ 0.942 ± 0.01; set/reset pulses (±1.0, ±3.0 for 30 steps) flip the sign and the state persists ≥ 250 steps with input off.
- `calibrate_gain(target_rho=0.95)` lands within ±0.02 on random nets.

### Phase 3 — Learning (engramnet-train, burn 0.22)
**Intent:** the three budget regimes and learnable topology as drop-in burn modules.
- `BudgetedLinear` (projection after optimizer step — option A; max-norm style), `FanOutWeightNorm` (w = g·v/‖v‖ along the *fan-out* axis — option B; implement the gradient as in weight normalisation but transposed), `GlobalBudget` (rescale ‖W‖_F after step — option C), `RigLRewire { drop_frac, grow_by: Gradient|Random, every }` (fixed edge count; drop smallest shares, grow by gradient magnitude), Hebbian/Oja local rules for the recurrent graph (no autodiff).
- Training recipe for recurrent graphs: BPTT over `relax` steps with `Leaky` update; gradient clipping; monitor ρ and σ_max every k steps (log via `tracing`).
**Tests:**
- Gradient check: `FanOutWeightNorm` backward vs finite differences (rel. err < 1e-4, f64 on `flex` backend).
- Invariant after every optimizer step for A/B/C (proptest over random batches).
- Golden (from `exp4_results.txt`): 2-layer tanh MLP on sklearn-digits-equivalent dataset vendored as `.safetensors` (1797×64, labels): base ≥ 0.97, A ≥ 0.97, B ≥ 0.97 test accuracy within 400 epochs, 3 seeds; B's learned g mean ∈ [1.0, 1.6].
- RigL: edge count constant across training (exact); accuracy within 2 points of dense at 80 % sparsity on the digits task.
- Backends: tests run on `flex`; `wgpu` job in CI is allowed to be skipped if no adapter is found; `cuda` is nightly-only.

### Phase 4 — Associative memory (engramnet-memory)
**Intent:** "select nodes → the network intuits the rest", with honest capacity numbers.
- `HebbianStore::new(n).store(&patterns)` → budgeted `Network` (fan-out projection after storage); `recall(cue: &[(NodeId, f32)], steps) -> RecallReport {overlap, converged}`; `Hopfield { patterns, beta }` modern-Hopfield one-step retrieval (softmax over stored patterns; returns the retrieved pattern and the softmax entropy as a "metastability" score); `capacity_sweep`.
**Golden tests (from `exp5_results.json`):** N=400, 20 trials: P=20 with 10 % cue ⇒ mean overlap ≥ 0.99; P=40 with 20 % cue ⇒ 0.80–0.93; P=56 with 20 % cue ⇒ < 0.70; cue corruption 20 % at P=40 ⇒ ≥ 0.60. Documented caveat test: un-budgeted store with explicit β=10 at P=40 ⇒ ≥ 0.99 (the budget is an automatic gain ≈ √(N/P), not a capacity gain). Hopfield: 2000 random unit patterns in d=256 retrieved exactly from 70 % cues with β=20 (≥ 0.99 cosine).

### Phase 5 — Engram table (engramnet-engram)
**Intent:** O(1), deterministic, prefetchable, editable context→cue lookup.
- 5a `Canonicalizer` (NFKC, lowercase, user merge table) over the `tokenizers` crate; `NgramKeyer { max_n, heads }` with multiplicative-XOR hash per (n, head) into prime-sized tables; `EngramTable { rows: Vec<Row{key_vec: UnitVec<d>, cue: Cue, meta}> }` persisted as safetensors (`key_vecs [M,d]`, `cue_idx`, `cue_val`, `row_meta.json`); `insert/remove/get`; collision policy (chain up to c entries per slot; LRU eviction); tiered cache (HBM/RAM/NVMe analogue: in-memory LRU + mmap'd file + optional disk).
- 5b (feature `ann`) learned-key retrieval via `usearch` for product-key-style memories.
- `Gate` with `Cosine{threshold}` and `Sigmoid{tau}`; `prefetch(tokens[..t+lookahead])` API.
**Tests:**
- Hash determinism across runs/platforms (fixed test vectors); collision rate at load factor 0.5 ≈ theoretical (±10 %); prime table sizes verified.
- Gating: synthetic rows with key_vecs at known angles ⇒ α monotone in cosine; mismatched (cos < 0) ⇒ α < 0.05.
- Round-trip persistence; fuzz `fuzz_load_table`.
- Bench: lookup for N=3, K=8 heads < 2 µs/token on CPU; gating for 64 candidates × d=256 < 20 µs.

### Phase 6 — Runner (engramnet-runner)
**Intent:** the "specialised decoder" loop with full traces.
- `Runner { table, gate, net, policy: InjectPolicy { mode, dose, hold, top_k }, relax_steps }`, `Runner::step(token) -> StepTrace`, `Runner::run(tokens) -> Vec<StepTrace>`, readout heads (`Readout::Linear` to a vocabulary or label), trace writer (MessagePack), replay tool.
- Conditional policies: `AlwaysGate` (default), `Allowlist(keys)`, `Never` (ablation only, test-gated behind `cfg(test)`/`--unsafe-ablation` flag).
**Tests (synthetic end-to-end):**
- Build a table mapping 50 synthetic 3-token contexts to 50 stored patterns in a N=400 budgeted Hebbian net; stream sequences; require ≥ 95 % correct pattern readout when the context matches and ≤ 5 % false injections (α > 0.5) when it does not. Ablation `Never`-gate must raise false injections (asserts the gate matters).
- Trace replay reproduces states bit-exactly with the same seed.
- Bench: end-to-end token step (lookup + gate + 20 relax steps, N=4096) < 20 ms CPU.

### Phase 7 — Weight-space programming interface (engramnet-wsp)
**Intent:** make the graph and table something a person or an LLM can read and safely edit.
- `export_readable(&Network, &EngramTable) -> ReadableGraph` (JSON: per node: budget g, out-shares w²/g² sorted, top-k targets with signs, in-degree, key direction if any; per table row: key summary, cue, meta); `Patch { ops: SetShares, AddEdge, RemoveEdge, SetBudget, TableInsert, TableRemove, ClampNode }` with `apply(&mut net, &table, &patch) -> Result<AuditRecord, Violation>`; blast-radius guard: computes max over driven nodes of ‖Δw_i‖ and refuses if > `max_delta` (default 2.0, the provable per-step bound); audit log (append-only JSONL with hashes of before/after safetensors).
- Feature `agent`: `AgentHarness` that sends `ReadableGraph` + a natural-language spec to an OpenAI-compatible endpoint and parses a `Patch` back; dry-run by default; never auto-applies without `--apply`.
**Tests:** every `Patch` leaves invariants intact (proptest); guard rejects oversize edits; `insta` snapshots of `ReadableGraph`; agent harness tested with a mock server; "program a latch" integration test: a hand-written `Patch` builds `hub_spokes(4)` from an empty graph and the set/reset test from Phase 2 passes.

### Phase 8 — LLM adapter (conditional FFN-neuron injection)
**Intent:** apply P2/P4 to real transformers: drive FFN key-value neurons only when the context matches, with bounded doses — the gated, conditional successor of hand-picked levers.
- **v1 (Python, `adapters/hf-transformers`)**: `GatedNeuronSteerer(model, rows: List[Row])` registering a forward pre-hook on `mlp.down_proj` that adds Σ αₜ·doseₙ·eₙ to the hook input, where αₜ is a CAST-style gate on the residual stream at the row's layer; calibration of dose by the median residual norm (same normalisation as the user's model cards); export of a *baked* equivalent (bias = Σ Δₙ W_down[:, n]) with an equivalence test. Loads rows from `engramnet` tables (via the Python package).
- **v2 (Rust, feature `candle`)**: same for architectures in candle-transformers; Qwen3.5 via the `crane` crate only if its maintainers expose a forward hook; otherwise document as unsupported.
**Tests:** hook vs baked-bias equivalence on a tiny HF model (e.g., a 100M-class model available ungated): max |Δlogit| < 0.5 and identical top-1 on a fixed prompt set; gate off (α < 0.05) on ≥ 95 % of a benign prompt set when the condition is "about chemistry lab safety" and on (α > 0.5) on ≥ 90 % of matching prompts; no test downloads or runs models > 1B parameters by default (`ENGRAMNET_BIG_MODELS=1` to opt in).

### Phase 9 — Release, docs, community
**Intent:** crates on crates.io, wheels on PyPI, a book, and a reproducible paper-style report.
- `cargo-release` workspace publishing in dependency order; `maturin` wheels (manylinux, macOS arm64/x86_64, Windows) via `wheels.yml`; docs.rs features set; mdBook with chapters = phases plus "Theory" (edge of chaos, budgets, capacity) and "Responsible use"; model-card and table-card templates; benchmark page auto-generated from criterion JSON.
**Tests:** `cargo publish --dry-run` for every crate; wheel smoke test (`import engramnet; engramnet.Network.unit_outgoing(64, 1.0)`); book builds with `mdbook test`.

---

## 6. Cross-cutting requirements

- **Numerics:** f32 storage/compute; f64 for spectral analysis and gradient checks; all tolerances explicit; `approx::assert_relative_eq`.
- **Determinism:** every randomised API takes a `seed: u64`; CI tests run with fixed seeds; traces include seeds.
- **No unsafe** outside `engramnet-py` and `memmap2` wrappers (audited).
- **Error handling:** `thiserror` enums per crate; no panics in library code paths (`#![deny(clippy::unwrap_used)]`).
- **Feature flags:** `core: default=["dense"]`, `sparse`; `engram: default=[], "ann"`; `train: "wgpu"|"cuda"|"metal"|"flex"`; `wsp: "agent"`; `py`.
- **Performance budgets:** listed per phase; regressions > 10 % fail `bench.yml` (informational until v0.3).
- **Observability:** `tracing` spans per runner step; `metrics` counters for lookups, gate pass-rate, injections, relax steps.
- **Security:** safetensors only (no pickle/torch.load anywhere in Rust; Python adapter uses `safetensors` too); fuzzed parsers; size limits on headers and tables.

## 7. Responsible-use policy (ships in the repo as `RESPONSIBLE_USE.md`)
- The framework exposes **conditional, gated** node selection only; no unconditional steering API outside test-only ablations.
- Examples and tables in the repo use benign conditions (topic routing, factual recall, formatting). The repo must not ship refusal-bypass or persona-override recipes; the HF adapter's README states this and credits the neuron-steering methodology it generalises (Lyra / neuron-explorer) without redistributing those recipes.
- Every published table/graph carries a card (purpose, conditions, dose limits, evaluation, known failure modes). The blast-radius guard and audit log are on by default.

## 8. Risks and open questions (track as GitHub issues at bootstrap)
1. **Capacity:** budgeted Hebbian recall collapses near 0.14·N; the Hopfield selector mitigates but adds O(P·d). Decide default by benchmark in Phase 6.
2. **Hash keys are brittle to paraphrase.** Mitigation: learned-key ANN mode (5b) and the gate; measure false-negative rate on paraphrased contexts.
3. **Edge-of-chaos default.** `FanOut{g:1.0}` sits exactly at the transition (synchronous updates oscillate for 1 ≤ g ≲ 1.2). Default to `Leaky{alpha:0.3}` and `g=0.95` for reservoirs; document.
4. **Qwen3.5 in Rust** is not upstream in candle (open issue); Crane supports the hybrid GDN/attention architecture. v2 adapter is best-effort.
5. **Weight-space programming claims** rest on a Nov-2026 blog post with released logs but no peer review; keep the agent harness experimental and off by default.
6. **Burn 0.22 API churn** (backend generics removed; tch/ndarray deprecated): pin versions, keep `train` isolated so core users are unaffected.
7. **Name/trademark:** "Engram" is also DeepSeek's module name; `engramnet` is distinct but verify crate availability and add a NOTICE line.

## 9. Operating instructions for Claude Code
1. Work phase by phase; do not start Phase k+1 until Phase k's definition of done is green in CI.
2. Write the tests listed for a phase first (they may fail), then implement until green; keep golden JSON oracles untouched unless `scripts/regenerate_golden.py` is re-run with a justification in the PR.
3. Prefer the simplest correct implementation; optimise only against a criterion benchmark that exists.
4. Never add unconditional steering entry points; never download/run models > 1B parameters without the opt-in env var; never run the user's Qwen3.5 repos.
5. Every public item gets docs with a runnable doctest; every crate gets a README with one 10-line example.
6. When a design decision is not covered here, write an ADR in `docs/adr/` and proceed; link it in the PR.
7. Keep Python to the three sanctioned places (bindings, HF adapter, notebooks). If a Rust dependency is missing a capability (e.g., a model architecture), document it in `docs/adr/` and fall back to the Python adapter rather than vendoring libtorch.

## 10. Milestone summary

| Milestone | Phases | Exit artifact |
|---|---|---|
| M1 "Bounded graph" | 0–2 | `engramnet-core`, `-spectral`, `-motifs` on crates.io (0.1); book chapters 1–3; golden tests for edge of chaos and latch |
| M2 "Learns & remembers" | 3–4 | `-train`, `-memory` (0.2); digits benchmark; capacity page |
| M3 "Runner" | 5–6 | `-engram`, `-runner`, CLI (0.3); end-to-end synthetic demo; traces |
| M4 "Programmable" | 7 | `-wsp` (0.4); readable export; patches; latch-from-spec demo |
| M5 "Adapter + release" | 8–9 | Python wheels, HF adapter v1, model/table cards, v0.5 announcement |

---

### References (primary; see the two reports for the full list)
Burn 0.22 (burn.dev; tracel-ai/burn) · Candle (huggingface/candle) · Crane (lucasjinreal/Crane) · faer (JOSS 2026) · safetensors & tokenizers (Hugging Face) · Cheng et al. 2026 (DeepSeek Engram, arXiv:2601.07372) · Lee et al. 2025 (CAST, arXiv:2409.05907) · Ramsauer et al. 2021 (arXiv:2008.02217) · Geva et al. 2021 (EMNLP) · Berges et al. 2024 (arXiv:2412.09764) · Salimans & Kingma 2016 · Srivastava et al. 2014 · Kodryan et al. 2022 · Loshchilov et al. 2024 (nGPT) · Newhouse et al. 2025 · Glorot & Bengio 2010 · Sompolinsky, Crisanti & Sommers 1988 · Schoenholz et al. 2017 · Yildiz, Jaeger & Kiebel 2012 · Evci et al. 2020 (RigL) · Takehi et al. 2026 (weight-space programming) · Liu, Ramirez et al. 2012 (engram cells).
