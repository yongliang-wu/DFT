
<div align="center">

# *On the Generalization of SFT*: <br>A Reinforcement Learning Perspective with <br>Reward Rectification


<a href="http://arxiv.org/abs/2508.05629" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-DFT-red?logo=arxiv" height="25" />
</a>

<a href="https://huggingface.co/collections/Liang0223/dft-6892da5e421a56a8deb48c9f" target="_blank">
    <img alt="HF Model: Cambrian-1" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Huggingface-Models-ffc107?color=ffc107&logoColor=white" height="25" />
</a>

<div style="font-family: charter; text-align: center; margin: 0 auto;">
                    <a href="https://yongliang-wu.github.io/" class="author-link" target="_blank">Yongliang Wu*</a> &emsp;
                    <a href="https://scholar.google.com/citations?user=dHBNmSkAAAAJ" class="author-link" target="_blank">Yizhou Zhou*</a> &emsp;
                    <a href="https://scholar.google.com/citations?user=IH2wK1cAAAAJ" class="author-link" target="_blank">Zhou Ziheng</a> &emsp;
                    <a href="https://github.com/ForJadeForest" class="author-link" target="_blank">Yingzhe Peng</a> &emsp;
                    <br>
                    <a href="https://scholar.google.com/citations?user=fdwhd9gAAAAJ" class="author-link" target="_blank">Xinyu Ye</a> &emsp;
                    <a href="https://joyhuyy1412.github.io/" class="author-link" target="_blank">Xinting Hu</a> &emsp;
                    <a href="https://vitozhu04.github.io/" class="author-link" target="_blank">Wenbo Zhu</a> &emsp;
                    <a href="http://luqi.info/" class="author-link" target="_blank">Lu Qi</a> &emsp;
                    <a href="https://faculty.ucmerced.edu/mhyang/" class="author-link" target="_blank">Ming-Hsuan Yang</a> &emsp;
                    <a href="https://yxpalmweb.github.io/" class="author-link" target="_blank">Xu Yang</a> &emsp;
</div>

<br>
</div>

## 🌟 Thanks for the Feedback of Community

We are grateful for the many thoughtful comments and feedback from the community regarding DFT, ranging from discussions of related ideas to reports of its application in different scenarios. We have heard of both successes and failures when applying DFT, for instance in literary or financial tasks.

Here, we would like to clarify that we do not claim DFT can replace SFT in all cases, as noted in our limitations section:

> *“While our experiments demonstrate substantial gains from DFT on mathematical reasoning benchmarks, this evaluation is confined to math-focused and code-focused (will be released in next version) datasets and models up to 7 billion parameters.”*

---

Nonetheless, these less successful cases, as well as community discussions on platforms such as Zhihu or Xiao Hong Shu about the intuitive principles behind DFT, together with our own experimental experience, have prompted us to think more deeply about the conditions under which DFT works well, and why it may be less effective in other contexts.

All this feedback reminds us of a remark by computing pioneer Richard Hamming in *The Art of Doing Science and Engineering: Learning to Learn* (p.27), which we have slightly adapted:

> *“Almost everyone who opens up a new field does not really understand it the way the followers—or the critics—do.”*

---

We hope this work can contribute to renewed interest in exploring the interplay between SFT and RL, and in better understanding the factors that underlie both the successes and the limitations of methods like DFT. Looking ahead, we also welcome researchers who are interested in our work to improve DFT in some of the currently unsuccessful cases, or in leveraging the ideas to uncover other connections between RL algorithms and SFT, ultimately aiming to achieve RL-like benefits at the cost of SFT across a broader range of settings.

## 📰 News

* **\[2025.08.08]** We have released the training scripts, evaluation scripts, and model checkpoints.

## Abstract
We present a simple yet theoretically motivated improvement to Supervised Fine-Tuning (SFT) for the Large Language Model (LLM), addressing its limited generalization compared to reinforcement learning (RL). Through mathematical analysis, we reveal that standard SFT gradients implicitly encode a problematic reward structure that may severely restrict the generalization capabilities of model. To rectify this, we propose Dynamic Fine-Tuning (DFT), stabilizing gradient updates for each token by dynamically rescaling the objective function with the probability of this token. Remarkably, this single-line code change significantly outperforms standard SFT across multiple challenging benchmarks and base models, demonstrating greatly improved generalization. Additionally, our approach shows competitive results in offline RL settings, offering an effective yet simpler alternative. This work bridges theoretical insight and practical solutions, substantially advancing SFT performance.

## Code Implementation
DFT is a **one-line change** to standard SFT: scale each token’s loss by its predicted probability (detached to avoid gradient flow).

```python
loss = loss * torch.softmax(shift_logits, dim=-1).gather(1, shift_labels.unsqueeze(-1)).squeeze(-1).detach()
```

## ⚙️ Installation

Our codebase has been tested on H100 servers with the following environment:

* `python 3.10.0`
* `torch 2.6.0+cu124`

```bash
git clone https://github.com/yongliang-wu/DFT.git
cd DFT
```

### 🔧 Set Up Training Environment

```bash
conda create -n DFT python=3.10 -y
conda activate DFT
cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

## 🚀 Getting Started

### Step 1: Prepare Data

```bash
# Generate training data (optional: change --train_end to control volume)
python examples/data_preprocess/numina_cot.py --train_end 100000

# Generate evaluation data
python examples/data_preprocess/math_dataset.py
```

### Step 2: Launch Training

```bash
nproc_per_node=8
project_name=numina-cot

experiment_name=numina-cot-dft-qwen-2.5-math-1.5b
save_path=checkpoints/$experiment_name

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_dft_trainer \
    data.train_files=data/numina_cot/train.parquet \
    data.val_files=data/math500/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=256 \
    data.max_length=2048 \
    optim.lr=5e-5 \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-Math-1.5B \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true
```

### Step 3: Evaluation

To evaluate the trained model, please first follow the [Qwen2.5-Math repository](https://github.com/QwenLM/Qwen2.5-Math) to set up the evaluation environment.

```bash
# Select the prompt format matching your model
PROMPT_TYPE="qwen-boxed"
# PROMPT_TYPE="llama-base-boxed"
# PROMPT_TYPE="deepseek-math"

# Set available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Configure sampling settings
N_SAMPLING=16
TEMPERATURE=1

# Specify model and output directories
MODEL_NAME_OR_PATH=""  # e.g., checkpoints/your-model-name
OUTPUT_DIR=""          # e.g., outputs/eval_results

# Run evaluation
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $N_SAMPLING $TEMPERATURE
```

## Limitations
Based on our evaluations and community feedback, DFT performs strongly on tasks with non-deterministic solution trajectories—i.e., those that admit multiple valid reasoning paths—such as mathematical chain-of-thought (CoT) reasoning, solutions to highly complex coding problems, and multimodal reasoning with informative CoT. By contrast, its performance is weaker on tasks with a single, well-specified ground-truth answer, particularly when the associated CoT (if exists) is highly constrained and near-deterministic (low-entropy).

## Citation
If you find this paper valuable for your research or applications, we would appreciate it if you could cite our work:
```latex
@article{wu2025generalization,
  title={On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification},
  author={Wu, Yongliang and Zhou, Yizhou and Ziheng, Zhou and Peng, Yingzhe and Ye, Xinyu and Hu, Xinting and Zhu, Wenbo and Qi, Lu and Yang, Ming-Hsuan and Yang, Xu},
  journal={arXiv preprint arXiv:2508.05629},
  year={2025}
}
```

## Related Repositories
* [https://github.com/huggingface/trl](https://github.com/huggingface/trl): TRL supports DFT now; see [the SFT Trainer documentation](https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md).
* [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): LLaMA-Factory supports DFT now, check [this script](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/extras/dft/qwen2_full_sft.yaml).
* [https://github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift): ms-swift supports DFT now, check [this script](https://github.com/modelscope/ms-swift/blob/main/examples/train/full/dft.sh).
* [https://github.com/Lauorie/DFT](https://github.com/Lauorie/DFT): Reproduced the DFT method without using Verl.
* [https://github.com/volcengine/verl](https://github.com/volcengine/verl): Codebase used for training.
* [https://github.com/QwenLM/Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math): Codebase used for evaluation.
