<p align="center">
  <h1 align="center">SpecKV</h1>
  <h3 align="center">Adaptive Speculative Decoding with Compression-Aware Gamma Selection</h3>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.02888"><img src="https://img.shields.io/badge/arXiv-2605.02888-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/amorfati123/speckv/stargazers"><img src="https://img.shields.io/github/stars/amorfati123/speckv?style=social" alt="GitHub stars"></a>
</p>


---

**Every speculative decoding system uses a fixed speculation length. We show that the optimal choice depends on both the task and the compression level, and that the draft model already knows the answer.**

SpecKV is a lightweight adaptive controller that selects the speculation length (gamma) per step using signals extracted from the draft model at zero additional cost. On a Llama 3.2 1B/3B pair across three compression levels (FP16, INT8, NF4), SpecKV achieves **56.0% more tokens per speculation step** than the standard fixed gamma=4, with only **0.34ms overhead** per decision (< 0.5% of step time). The result is statistically significant at *p* < 0.001 (paired bootstrap, 10K resamples).

## The Core Idea

In standard speculative decoding, the draft model proposes gamma tokens and the target model verifies them. The number of tokens produced per step is:

$$\text{tokens} = k + 1, \quad k \in \{0, \ldots, \gamma\}$$

where *k* is the number of accepted tokens. The acceptance rate depends on how well the draft predicts the target. When the target model is compressed (quantized), its output distribution shifts, changing which gamma values are effective.

SpecKV observes that the draft model's own uncertainty signals predict this acceptance rate:

$$\hat{a} = f(\bar{H}, \bar{c}, H_{\max}, c_{\min}, \text{comp}, \gamma)$$

where the bar{H} is mean draft entropy, bar{c} is mean draft confidence, and the max/min variants capture the worst-case signal within a step. These features are already computed during speculation and are normally discarded. SpecKV retains them and feeds them to a small MLP (16 hidden units) that predicts acceptance rate for each candidate gamma. The policy then picks:

$$\gamma^* = \arg\max_{\gamma \in \{2, 4, 6, 8\}} \hat{a}(\gamma) \cdot \gamma + 1$$

That is it. No retraining, no extra forward passes, no architectural changes.

## Key Results

| Policy | Expected Tokens/Step | vs Fixed-4 |
|:---|:---:|:---:|
| Fixed-4 (default) | 3.73 | baseline |
| Fixed-best (per compression) | 5.81 | +55.8% |
| Task-oracle (requires task label) | 4.07 | +9.1% |
| **SpecKV-fast** | **5.82** | **+56.0%** |
| SpecKV-accurate (RF-100, high overhead) | 6.39 | +71.3% |

SpecKV-fast matches Fixed-best without needing any prior knowledge of which fixed gamma works best. It uses only per-step draft signals. Full breakdown:

| Compression | Task | Fixed-4 | SpecKV | Improvement |
|:---|:---|:---:|:---:|:---:|
| FP16 | Code | 3.95 | 6.32 | +60.0% |
| FP16 | Math | 4.17 | 6.74 | +61.7% |
| INT8 | Math | 4.29 | 7.05 | +64.5% |
| NF4 | Code | 4.02 | 6.47 | +60.9% |
| NF4 | Chat | 3.33 | 4.95 | +48.5% |

95% bootstrap CIs: SpecKV-fast [5.69, 5.96] vs Fixed-4 [3.67, 3.79]. Non-overlapping. *p* < 0.001.

## Why This Matters for Your Work

If you are working on any of the following, SpecKV is directly relevant:

- **Speculative decoding systems** (vLLM, SGLang, TensorRT-LLM): You can plug the adaptive gamma selection into your existing speculation loop. The only requirement is access to draft token probabilities, which you already compute.
- **Model compression for inference**: Our profiling data (5,112 step-level records across FP16/INT8/NF4) shows exactly how compression shifts acceptance rates. This data is useful even if you do not use the SpecKV controller.
- **Adaptive inference research**: The contextual bandit formulation and the finding that draft entropy/confidence predict acceptance rate opens up further work on online adaptation, tree-structured speculation, and joint optimization with KV cache eviction.

## What Is in This Repo

```
speckv/
  notebooks/
    Phase3_Adaptive_Controller.ipynb   # Trains the adaptive policy
    Phase4_Final_Evaluation.ipynb      # Final evaluation, CIs, paper tables
  results/
    phase1_profiling_results.csv       # Throughput vs gamma (FP16 only, vLLM)
    phase2_experiment_results.csv      # 240 experiments: compression x gamma x task
    phase2_optimal_gamma.csv           # Optimal gamma per task per compression
    phase3_policy_comparison.csv       # Policy simulation results
    phase3_predictor_comparison.csv    # Predictor architecture comparison
    phase3_improvement.csv             # SpecKV improvement breakdown
    paper_table1_policy_comparison.csv # Table 1 from the paper
    paper_table2_detailed_breakdown.csv# Table 2 from the paper
    paper_confidence_intervals.csv     # Bootstrap CIs
    paper_numbers.json                 # All key numbers in one file
```

## Reproducing the Results

**Requirements**: Python 3.9+, PyTorch 2.6+, scikit-learn, pandas, matplotlib, seaborn. No GPU needed for Phase 3 and Phase 4 (they run on the CSV data from Phase 2).

```bash
git clone https://github.com/amorfati123/speckv.git
cd speckv
pip install pandas scikit-learn matplotlib seaborn
```

Open `notebooks/Phase3_Adaptive_Controller.ipynb` and run all cells. It trains the predictor on the Phase 2 step data and evaluates all policies.

Open `notebooks/Phase4_Final_Evaluation.ipynb` to reproduce the paper tables, confidence intervals, and all figures.

To reproduce the profiling data from scratch (requires an NVIDIA GPU with 24GB+ VRAM):

```bash
pip install vllm transformers bitsandbytes
# Phase 1: vLLM-based throughput profiling
# Phase 2: Manual speculative decoding with per-step logging
# See the paper for full experimental details
```

## Using SpecKV in Your Own Pipeline

The core logic is simple. After each draft step, extract entropy and confidence from the draft token probabilities, then query the predictor for each candidate gamma:

```python
import numpy as np
from sklearn.neural_network import MLPRegressor

# train the predictor on your own profiling data (or use ours)
predictor = MLPRegressor(hidden_layer_sizes=(16,), max_iter=500)
predictor.fit(X_train, y_train)  # features -> acceptance_rate

# at inference time, per speculation step:
def select_gamma(draft_entropy, draft_confidence, max_entropy, min_confidence, comp_enc):
    best_gamma, best_expected = 2, 0
    features_base = [draft_entropy, draft_confidence, max_entropy, min_confidence, comp_enc]
    for gamma in [2, 4, 6, 8]:
        features = np.array(features_base + [gamma]).reshape(1, -1)
        predicted_ar = np.clip(predictor.predict(features)[0], 0, 1)
        expected_tokens = predicted_ar * gamma + 1
        if expected_tokens > best_expected:
            best_expected = expected_tokens
            best_gamma = gamma
    return best_gamma
```

The entire decision takes 0.34ms on CPU. No GPU needed for the controller.

## Feature Importance

What drives the gamma decision? The draft model's worst-case signals within a step matter most:

| Feature | Importance |
|:---|:---:|
| min_draft_confidence | 30.0% |
| max_draft_entropy | 24.1% |
| mean_draft_confidence | 21.8% |
| mean_draft_entropy | 17.7% |
| gamma | 3.4% |
| compression_level | 3.1% |

The draft model's own uncertainty is the dominant signal. Compression level and gamma are minor factors, meaning the policy generalizes across configurations.

## Limitations and Ongoing Work

We are transparent about the current limitations:

- **Model scale**: Evaluated on 1B/3B. Scaling to 8B/70B is in progress.
- **Prompt diversity**: 20 prompts across 4 tasks. Scaling to HumanEval (164), GSM8K (1,319), and ShareGPT is in progress.
- **Simulation-based evaluation**: Results are from offline policy evaluation. End-to-end integration into the HuggingFace generation loop is in progress.

These are engineering tasks, not conceptual gaps. The core finding (compression shifts optimal gamma, draft signals predict acceptance rate, adaptive selection improves throughput) is robust.

## Citation

If you use SpecKV in your research, please cite:

```bibtex
@article{shukla2026speckv,
  title={SpecKV: Adaptive Speculative Decoding with Compression-Aware Gamma Selection},
  author={Shukla, Shikhar},
  journal={arXiv preprint arXiv:2605.02888},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built with a single RTX 3090 and a lot of patience.
