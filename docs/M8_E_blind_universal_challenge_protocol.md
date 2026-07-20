# M8-E: Blind Universal Concept Challenge Protocol

This document establishes the scientific protocol for testing the extreme limit of the Latent Concept Intermediate Representation (LCIR). We aim to falsify the hypothesis that LCIR functions as a true intermediate compilation layer.

## 1. Hypothesis

**H0 (Null Hypothesis)**: The LCIR is merely a descriptive abstraction. It lacks the sufficient mathematical density to be blindly compiled onto unseen models without pre-existing parallel data or architectural leakage. Compiling blindly will fail causal preservation.
**H1 (Alternative Hypothesis)**: LCIR is a functional intermediate representation (FIR). It correctly abstracts behavior independent of target architecture and can be compiled into causal geometric vectors for entirely opaque models.

---

## 2. Validation Tests

### A. Blind Target Model Test
The Encoder is trained strictly on `Qwen` and `Phi`. The Backend Compiler targets an entirely *opaque* target model (e.g., a simulated BlackBox-7B model).
**Constraints**:
- The backend must not use parallel hidden states.
- The backend must not use pre-computed concept vectors from the target.
- It relies purely on the mathematical compilation of the LCIR semantic constraints.
**Metrics**:
- Causal Effect Preservation (Compiled vs Simulated Native)
- Cohen's $d$
- Bootstrap Confidence Intervals (95%)
- Permutation $p$-value

### B. LCIR Inversion Test
Tests the mathematical losslessness of the abstraction.
1. Extract Native Concepts $\rightarrow$ Universal Encoder $\rightarrow$ Original LCIR
2. Compile Original LCIR $\rightarrow$ Target Models $\rightarrow$ Compiled Concepts
3. Re-Encode Compiled Concepts $\rightarrow$ Universal Encoder $\rightarrow$ Reconstructed LCIR
**Goal**: Measure the divergence between Original LCIR and Reconstructed LCIR semantic dimensions.

### C. LCIR Composition Test
Tests the compilation of composed behavioral abstractions.
1. Define a complex LCIR combining: `Authority` + `Planning` + `Uncertainty`.
2. Compile as a single behavioral instruction.
**Metrics**:
- Non-linear interference score.
- Causal preservation of all sub-components.
- Semantic consistency.

---

## 3. Final Verdict Classification

The suite calculates the overall success across the three tests and categorizes the LCIR framework:

- **LCIR_DESCRIPTIVE_ONLY**: If the Blind Target Model test fails statistical significance (Cohen's $d < 0.2$, $p > 0.05$) OR if Composition suffers from severe non-linear manifold collapse. LCIR is a useful tag, but not a compiler.
- **LCIR_FUNCTIONAL_INTERMEDIATE_REPRESENTATION**: All tests pass. The framework successfully translates abstract behaviors into functional geometric constraints blindly, and allows reliable composition of abstractions.
