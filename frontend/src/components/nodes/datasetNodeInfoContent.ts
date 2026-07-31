/**
 * Rich descriptions for dataset-category nodes. Use $$...$$ for display math and $...$ for inline math (KaTeX).
 * Paragraphs are separated by a blank line (\n\n).
 */
export const DATASET_NODE_INFO_MARKDOWN = {
  linear_dataset: `Synthetic linear regression for the MSE trainer. Each sample draws a random matrix $W$ and noise, then forms
$$y = Wx + \\sigma\\varepsilon$$
with $W_{ij}\\sim\\mathcal{N}(0,1)/\\sqrt{d_{\\mathrm{in}}}$ and i.i.d. $\\varepsilon$. Comma-separated parameter lists run as a Cartesian sweep.`,

  random_noise_dataset: `Independent random vectors for both endpoints. For each sample, draw
$$x\\in\\mathbb{R}^{d_{\\mathrm{in}}},\\quad y\\in\\mathbb{R}^{d_{\\mathrm{out}}}$$
from the selected input distribution, independently across samples and between $x$ and $y$.`,

  memorization_a_dataset: `Memorization benchmark dataset: draw random continuous inputs $x\\in\\mathbb{R}^{d_{\\mathrm{in}}}$, then sample class labels independently of $x$. Output classes follow a configurable prior:
* uniform: $P(c_n)=1/d_{\\mathrm{out}}$
* power law: $P(c_n)\\propto 1/n^{\\alpha}$
* exponential: $P(c_n)\\propto e^{-\\alpha n}$
Use **cross_entropy_loss** with this node (the trainer rejects **mse_loss** here). Cross-entropy measures pure memorization capacity.`,

  memorization_b_dataset: `Memorization B (categorical inputs): use one shared vocabulary size $V$ (node field: **vocab size**). Sample an input class $c_{\\mathrm{in}}\\in\\{0,\\ldots,V-1\\}$ and an output class $c_{\\mathrm{out}}\\in\\{0,\\ldots,V-1\\}$ **independently**, both from the same prior family as memorization A (uniform / power law / exponential with shared $\\alpha$). Dense MLP path uses one-hot rows in $\\mathbb{R}^{V}$; token MLP path uses token ids with width $1$.`,

  symbolic_func_dataset: `Define $y(x)$ with LaTeX; SymPy on the server compiles the expression together with numeric scalar parameters. The node emits sampled $(x,y)$ pairs for regression. Preview KaTeX in the node body; runtime parsing may accept slightly broader syntax than the preview.

Output dim is fixed to $1$ (scalar $y$) in v1.

Use symbols $x_1\\ldots x_d$ (where $d$ is input dim), summation $\\sum_{i=1}^{d}$, and numeric constants directly (for example $10x_i$), or $\\pi$ for the constant pi. Server compilation requires \\texttt{antlr4-python3-runtime==4.11.1}.`,

  token_prediction_dataset: `Random token sequences over a vocabulary. Choose retrieval by position (target at a fixed index) or retrieval by content (target follows the nearest prior occurrence of the query token).

* position mode: retrieval target uses Python indexing via $\\texttt{which token}$
* content mode: retrieval target is the nearest prior token to the last token (by absolute token distance)

Useful for probing next-token models beyond simple i.i.d. unigrams; typically pair with an attention layer plus cross-entropy loss.

**References:** [Physics of LLMs — Part 3.1 (knowledge / in-context)](https://arxiv.org/abs/2309.14316), [series hub](https://physics.allen-zhu.com/).`,

  circle_random_walk_dataset: `A bigram random walk on a circular vocabulary $\\{0,\\ldots,V-1\\}$. From token $x$, the next token steps left or right with configurable bias, wrapping modulo $V$.`,

  circular_motion_dataset: `Toy planar circular motion at discrete timesteps. Each timestep is two token ids $(t_x, t_y)$ from the same vocabulary $\\{0,\\ldots,V-1\\}$ (quantized coordinates). Context is $[\\texttt{batch}, L, 2]$; the target is the next timestep pair. Use **Transformer (multiple tokens)** with $\\texttt{tokensPerPosition}=2$.`,

  unigram_dataset: `i.i.d. token sequences over $\\{0,\\ldots,V-1\\}$ with a configurable output distribution over ranks $i\\in\\{1,\\ldots,V\\}$ (same shapes as memorization A class priors):
* uniform: $P(i)=1/V$
* power law: $P(i)\\propto i^{-\\alpha}$
* exponential: $P(i)\\propto e^{-\\alpha i}$
Draw i.i.d. context tokens and next-token targets from the same normalized law.`,

  bigram_low_rank_dataset: `Bigram data with a low-rank factorization of transition logits. Columns of the left factor are scaled by $\\lambda_n$ with $n=1\\ldots R$: either $\\lambda_n=n^{-\\alpha}$ (power law) or $\\lambda_n=e^{-\\alpha n}$ (exponential). Sample $x$ from the stationary distribution, then $y\\sim \\mathrm{softmax}(\\mathrm{low\\text{-}rank}(x))$.`,

  random_input_distribution: `Draws random inputs $x$ from a base distribution plus optional Gaussian jitter. Wire the output into an Input sampler; sample count lives on the sampler node.`,

  input_sampler: `Samples an input tensor from a wired **Random input distribution**. Set the sample count on this node, then connect its **sample tensor** output into teacher-dataset train/test input sockets.`,

  teacher_dataset: `Knowledge-distillation style pairs $y = f_{\\mathrm{teacher}}(x)$. Connect a teacher model (often an MLP), then wire **train input** and **test input** from input samplers (sample tensors). One **dataset** output wires the whole node into a trainer (train and test sizes live in the node’s parameters).`,

  in_context_associative_recall_dataset: `Sequences of key–value pairs followed by a query key; the supervision target is the value associated with that key (associative recall in context).`,

  uniform_linear_motion_dataset: `Uniform linear motion in $D$ dimensions over discrete time. With position $x_0$ and velocity $v$, samples follow
$$x_i = x_0 + v\\cdot i$$
Tensors are shaped $[\\mathrm{batch},T,D]$: inputs are $x_0\\ldots x_{T-1}$ and targets $x_1\\ldots x_T$. Optional noise can jitter only the final target row $x_T$.

For one-step MSE setups, use loss mask \`last context slot only\`.`,

  modular_addition_dataset: `Modular arithmetic classification over pairs of tokens. Inputs are ordered pairs $(a,b)$ with $a,b\\in\\{0,\\ldots,p-1\\}$ and target
$$y=(a+b)\\bmod p.$$
The node builds the full grid of $p^2$ pairs, shuffles with the seed, and takes the first $\\mathrm{round}(f\\cdot p^2)$ rows as train (clamped to at least one); the remaining rows are the test split.`,

  dataset_mixer: `Mixes two upstream datasets with the same input/output dimensions for shared-model training; one **dataset** output carries both configured train and test sample counts.

Set **total train samples** $N_{\\mathrm{train}}$ and **total test samples** $N_{\\mathrm{test}}$, plus dataset A ratio $p_A$ (with $p_B=1-p_A$). For each stream the node draws
$$N_A=\\mathrm{round}(N\\cdot p_A),\\quad N_B=N-N_A$$
samples from A/B respectively, concatenates along the batch dimension, then shuffles with **init seed**.`,

  dataset_mixer_a: `Dataset Mixer A is the original concatenation mixer. It samples from dataset A and dataset B by ratio, concatenates the resulting batches, then shuffles.`,

  dataset_mixer_b: `Dataset Mixer B uses **one** feature batch $x$ per row, then mixes outputs. **Train** and **test** each draw $x$ from **dataset A**'s input law (same count as A's train/test sizes); **dataset B** is always evaluated on that same $x$ (so B's train and test inputs match A's). Both branches must declare matching input/output dims and train/test sizes.

$$O_\\lambda = \\lambda O_1(x) + (1-\\lambda)O_2(x)$$`,

  pcfg_dataset: `**Synthetic PCFG-style token LM** with two generators (\\texttt{pcfgGenMode}):

- **\\texttt{binary_tree}** (default): legacy random full binary tree whose leaves are i.i.d. integers in the vocab — good for plumbing / entropy-floor baselines; not a literal weighted CFG over words.
- **\\texttt{cfg_sentence}**: NLTK-style **weighted productions** from a small builtin grammar (\\texttt{pcfgGrammarId}=\\texttt{world_model}: S $\\rightarrow$ NP VP, NP $\\rightarrow$ Adj N $|$ N, VP $\\rightarrow$ V NP, fixed lexicon). Terminals map to stable low ids; **\\texttt{vocabSize}** is raised to at least the number of terminals so ids stay in-range. **\\texttt{pcfgMaxDepth}** scales a safety cap on expansion depth (not the binary-tree knobs). Sequences are still padded/truncated to $\\texttt{contextLength}+1$ for next-token LM.

Controllable-structure motivation: PCFGs as synthetic corpora for mechanistic / training-dynamics work (e.g. hierarchical latent generation). Minimal external pattern: [thomasbreydo/pcfg](https://github.com/thomasbreydo/pcfg) (NLTK-style \\texttt{generate}). See also [Physics of LLMs — Part 1 (CFG)](https://arxiv.org/abs/2305.13673), [Johnson 1998 (PCFG)](https://aclanthology.org/J98-4004/).`,

  dyck_dataset: `**Dyck / balanced bracket strings** mapped to token ids. **\\texttt{vocabSize}** is fixed to **$2k$** where **\\texttt{numBracketTypes}** $=k$: each bracket type uses one open id ($0,2,\\ldots$) and one close id ($1,3,\\ldots$). A full window has length **\\texttt{contextLength}+1**; **train input** and **train output** are both $N\\times L$ with $L=\\texttt{contextLength}$: row $x_i$ is tokens $s_0\\ldots s_{L-1}$ and row $y_i$ is the one-step-ahead targets $s_1\\ldots s_L$ (standard causal LM slice). Token sequence models emit **per-position logits** $N\\times L\\times V$ for cross-entropy over those targets. **\\texttt{maxNestingDepth}** (optional): when $>0$, limits how many unmatched opens may sit on the stack while sampling the balanced prefix; **0** leaves depth implicit (still $\\leq$ context length).

References: [Suzgun et al. 2019](https://aclanthology.org/W19-3905/), [Hahn 2020](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00306/96489).`,

  ngram_language_dataset: `**Order-$n$ Markov tokens** with Dirichlet-smoothed random transition rows (table RNG keyed by seed).

Classic sequential modelling refs: Shannon (1948), Jelinek (1997) — see node bibliography in docs if needed.`,

  formal_language_suite_dataset: `Controlled generators for **$a^n b^n$**, **$a^n b^n c^n$**, **palindromes**, and **parity/XOR** bits; uses vocab ids $0,1,\\ldots$ with padding.

Related classical complexity framing: [Weiss et al. 2018](https://arxiv.org/abs/1805.04908).`,

  scan_dataset: `SCAN-like command $\\rightarrow$ action toy data. **Synthetic** mode samples template commands; **download** mode tries to fetch [Lake \\& Baroni SCAN tasks](https://arxiv.org/abs/1711.00350) (\`tasks.txt\`) into the cache dir.

**Vocab size $v$:** same id-hash idea as COGS toy—$v$ is the LM’s token-id modulus, not the count of distinct command words in the template pools.

Paper: [Lake \\& Baroni 2018](https://arxiv.org/abs/1711.00350).`,

  cogs_dataset: `Synthetic semantic-paraphrase style pairs (sentence tokens $\\rightarrow$ lightweight logical-form proxy token). Intended as a **controlled** COGS-flavored probe; not the full EMNLP release.

**Vocab size $v$:** controls the **integer token range** used by the LM (ids are $1 + (\\texttt{hash}(\\text{word}) \\bmod (v-2))$, padding $0$). The **readable** template uses a **small fixed English lexicon** (a few nouns, verbs, determiners)—raising $v$ does **not** grow that word list; it only changes how surface words map to ids (and collision structure).

Reference style: [Kim \\& Linzen 2020](https://aclanthology.org/2020.emnlp-main.731/).`,

  listops_dataset: `Toy **[ MIN $a$ $b$ ]**-style integer expressions padded to context; target is $\\min(a,b)$ modulo vocab.

Reference: [Nangia \\& Bowman 2018](https://aclanthology.org/W18-5432/).`,

  tinystories_dataset: `**Synthetic TinyStories-style** short prose when offline; optional **download** if you supply a small \`.txt\` URL (\`tinyStoriesUrl\`). Disclosure: synthetic mode is **not** the Eldan \\& Li corpus.

Paper: [Eldan \\& Li 2023](https://arxiv.org/abs/2305.07759).`,

  phi1_style_dataset: `**Synthetic textbook / QA / tiny-code** mixes inspired by high-quality training narratives around phi-style setups; **not** Microsoft’s proprietary mix.

Paper: [Gunasekar et al. 2023](https://arxiv.org/abs/2306.11644).`,

  biography_lm_dataset: `**Synthetic biography / fact-sheet token LM** (offline only). Structured blocks over reserved low vocab ids plus entity slots; supports fixed template order, shuffled field blocks, or noisy entity tokens—useful for probing fact storage vs augmentation.

**References:** [Physics of LLMs — Part 3.1 (knowledge storage)](https://arxiv.org/abs/2309.14316), [series hub](https://physics.allen-zhu.com/).`,

  relation_tuple_dataset: `**Synthetic $(\\mathrm{subject}, \\mathrm{relation}, \\mathrm{object})$ sequences** as next-token LM. **Forward** mode emits a labeled triple; **inverse** mode poses a query $(\\mathrm{OBJ}, \\mathrm{REL})$ then reveals the subject token—tests relational retrieval pressure without real KG text.

**References:** [Physics of LLMs — Part 3.1 (knowledge / relations)](https://arxiv.org/abs/2309.14316), [series hub](https://physics.allen-zhu.com/).`,

  synthetic_playground_dataset: `**Single-node playground presets** isolating capability axes (local parity / copy trace / modular state / nested brackets / short fact snippets). All generators are seed-controlled synthetic streams mapped to configurable vocab and context length.

Names echo instrument-style toy families from Physics-of-LLMs narratives; implementations here are ComfyResearch-native approximations for rapid sweeps.

**References:** [Physics of LLMs — Part 1 (CFG / structure)](https://arxiv.org/abs/2305.13673), [Part 3.1 (knowledge)](https://arxiv.org/abs/2309.14316), [series hub](https://physics.allen-zhu.com/).`,

  multi_hop_fact_chain_dataset: `**Linear chains of synthetic binary facts** followed by a short query segment; next-token supervision follows the usual $\\texttt{contextLength}+1$ windowing. Controls **chain hops** to scale relational depth before the query block.

**References:** [Physics of LLMs — Part 3.1 (multi-hop / knowledge depth)](https://arxiv.org/abs/2309.14316), [series hub](https://physics.allen-zhu.com/).`,

  mnist_dataset: `Official **MNIST** digits: grayscale tensors $[N,1,28,28]$ in $[0,1]$, labels $0\\ldots 9$. Downloads the standard IDX archives over HTTPS and caches them under **download cache dir** (empty uses \`~/.cache/comfy_research_mnist\`). For an offline blob toy with tunable noise, use **Gaussian blob dataset** instead. Pair with **resnet_model** or **vit_model** and **cross_entropy_loss**.`,

  gaussian_blob_dataset: `**Multi-class** toy vision data on $[N,1,H,H]$ tensors ($H$ configurable, default 28). **# classes** sets $K$ with labels $0\\ldots K-1$; each class is a smooth **2D Gaussian blob** at a grid cell inside the canvas. **noise level** adds per-pixel Gaussian noise (prototype jitter scales with the same knob). Generated locally—useful for MNIST-shaped training loops without IDX downloads or for quick MLP/ResNet sanity checks.`,

  shape_world_dataset: `Three-way classification on synthetic $[N,1,H,W]$ images: filled **square**, **triangle**, or **circle** on a gray field with **adjustable noise level** (additive Gaussian after drawing). Useful as a minimal compositional vision prior beyond MNIST digits.`,

  hole_counting_dataset: `Count foreground **holes** (circular erasures) in a synthetic blob; labels are integers $0\\ldots k$ where $k$ is **max holes**. Images are $[N,1,H,H]$ float in $[0,1]$.`,

  diffusion_pde_dataset: `**2D periodic heat equation** on an $H\\times H$ grid with explicit Euler updates:
$$u_{t+1} = u_t + \\Delta t\\, D\\, \\Delta u_t$$
Each sample draws a random Gaussian initial field, warms up for \\texttt{warmupSteps}, then records $T=$ \\texttt{contextFrames} consecutive snapshots. Flattened tensors have length $T\\cdot C\\cdot H^2$ for input (frames $0\\ldots T{-}1$) and the same for target (frames $1\\ldots T$).

Pair with **MPP-style spatiotemporal ViT** or an MLP on the flattened vector. [MPP paper](https://arxiv.org/abs/2310.02994).`,

  reaction_diffusion_dataset: `**Fisher–KPP reaction–diffusion** on a periodic $H\\times H$ lattice:
$$u_{t+1} = u_t + \\Delta t\\left(D\\,\\Delta u_t + r\\,u_t(1-u_t)\\right)$$
Same flattening convention as the diffusion PDE dataset ($TCHW$ windows). Keep $\\Delta t$ moderate for stability.

[MPP](https://arxiv.org/abs/2310.02994); code patterns echo [multiple_physics_pretraining](https://github.com/PolymathicAI/multiple_physics_pretraining).`,

  advection_dataset: `**2D constant-coefficient advection** with periodic boundaries and central spatial differences:
$$u_{t+1} = u_t - \\Delta t\\left(v_x\\,\\partial_x u + v_y\\,\\partial_y u\\right)$$
Snapshots are stacked and flattened like the other field datasets (length $TCHW$).

[MPP](https://arxiv.org/abs/2310.02994).`,






} as const;

export type DatasetNodeInfoKind = keyof typeof DATASET_NODE_INFO_MARKDOWN;

export function datasetNodeInfoTitle(kind: DatasetNodeInfoKind): string {
  const labels: Record<DatasetNodeInfoKind, string> = {
    linear_dataset: "Linear dataset",
    random_noise_dataset: "Random noise dataset",
    memorization_a_dataset: "Memorization A dataset",
    memorization_b_dataset: "Memorization B dataset",
    symbolic_func_dataset: "Symbolic function dataset",
    token_prediction_dataset: "Token Retrieval dataset",
    circle_random_walk_dataset: "Circle random walk dataset",
    circular_motion_dataset: "Circular motion dataset",
    unigram_dataset: "Unigram dataset",
    bigram_low_rank_dataset: "Bigram low-rank dataset",
    random_input_distribution: "Random input distribution",
    input_sampler: "Input sampler",
    teacher_dataset: "Teacher dataset",
    in_context_associative_recall_dataset: "In-context associative recall",
    uniform_linear_motion_dataset: "Uniform linear motion dataset",
    modular_addition_dataset: "Modular addition dataset",
    dataset_mixer: "Dataset mixer",
    dataset_mixer_a: "Dataset mixer A",
    dataset_mixer_b: "Dataset mixer B",
    pcfg_dataset: "PCFG toy LM dataset",
    dyck_dataset: "Dyck language dataset",
    ngram_language_dataset: "N-gram language dataset",
    formal_language_suite_dataset: "Formal language suite dataset",
    scan_dataset: "SCAN toy dataset",
    cogs_dataset: "COGS-style toy dataset",
    listops_dataset: "ListOps toy dataset",
    tinystories_dataset: "TinyStories-style corpus dataset",
    phi1_style_dataset: "phi-1 style corpus dataset",
    biography_lm_dataset: "Biography / fact-sheet synthetic LM",
    relation_tuple_dataset: "Relation tuple synthetic LM",
    synthetic_playground_dataset: "Synthetic playground LM",
    multi_hop_fact_chain_dataset: "Multi-hop fact chain synthetic LM",
    mnist_dataset: "MNIST (official IDX)",
    gaussian_blob_dataset: "Gaussian blob dataset",
    shape_world_dataset: "Shape world dataset",
    hole_counting_dataset: "Hole counting dataset",
    diffusion_pde_dataset: "Diffusion PDE field dataset",
    reaction_diffusion_dataset: "Reaction–diffusion field dataset",
    advection_dataset: "Advection field dataset",
  };
  return labels[kind];
}
