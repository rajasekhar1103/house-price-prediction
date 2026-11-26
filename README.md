# Sarcasm Dataset and Qwen3 Experiments

This repository contains notebooks and scripts for generating sarcastic
conversation examples, downloading a Reddit sarcasm dataset, and experimenting
with Qwen3-4B fine-tuning.

## Workflows

| Workflow | File | Purpose |
| --- | --- | --- |
| Generate a conversational dataset | [Sarcastic_Dataset_Generator_6000.ipynb](Sarcastic_Dataset_Generator_6000.ipynb) | Generate and validate sarcastic Q&A examples with NVIDIA NIM. The notebook runs in Kaggle and makes billable API requests. |
| Download the Reddit dataset | [sample_ml_colab.ipynb](sample_ml_colab.ipynb) | Download the upstream Reddit sarcasm dataset in Colab. The dataset is large and is not stored in this repository. |
| Fine-tune Qwen3-4B | [reddit_sarcasm_qwen3_colab.ipynb](reddit_sarcasm_qwen3_colab.ipynb) | Prepare the Reddit data and train a LoRA adapter in Kaggle or Colab. A GPU is required. |
| Load Qwen3-4B | [qwen3_4b_unsloth_colab.ipynb](qwen3_4b_unsloth_colab.ipynb) | Download and run the base model in Colab. A GPU is required. |
| Generate sarcastic examples from a script | [nvidia_image_response.py](nvidia_image_response.py) | Run the NVIDIA API-backed generation utility. |

## Dataset files

- `dataset_4000.jsonl` and `dataset.jsonl` are separate JSONL exports. Check
  their provenance and intended version before using either for training.
- The notebooks save generated datasets, checkpoints, and model adapters to
  their runtime's working/output directory; those files are not automatically
  part of this repository.
- Keep large downloaded datasets and model weights out of Git. Use the
  notebook/runtime storage or an appropriate dataset/model hosting service.

## Local setup

The local Python utility uses the packages in `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Provide `NVIDIA_API_KEY` through the environment or a local `.env` file when
using NVIDIA-backed generation. Never commit API keys. The notebook workflows
install their additional training dependencies in the notebook runtime.

## Repository hygiene

Virtual environments, notebook checkpoints, local Kaggle output copies, logs,
and environment files are excluded by `.gitignore`. Review generated datasets
before adding them to GitHub, especially for duplicates, malformed rows, and
potentially sensitive content.
