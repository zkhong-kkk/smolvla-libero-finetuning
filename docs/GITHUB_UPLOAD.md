# GitHub Upload Workflow

The repository can be pushed either from the local computer or directly from
the training server. The recommended workflow is:

1. Keep source code, documentation, small logs, CSV results, and selected short
   videos in Git.
2. Keep checkpoints and optimizer states outside normal Git.
3. Publish model weights to Hugging Face Hub if they need to be shared.
4. Never commit access tokens, private SSH keys, `.env` files, or Hugging Face
   credentials.

The final Stage 2 checkpoint is hosted at
[zkhomh-kkk/smolvla-libero-spatial-stage2](https://huggingface.co/zkhomh-kkk/smolvla-libero-spatial-stage2).
Only the link and small checkpoint metadata belong in this Git repository.

## Before the First Push

```bash
git status
git check-ignore -v path/to/model.safetensors
git add .
git status
git commit -m "Add SmolVLA LIBERO fine-tuning project"
```

Inspect `git status` before committing. No `.safetensors`, optimizer states,
datasets, cache directories, or bulk videos should appear.

## Large Model Checkpoint

Do not bypass GitHub's file-size limits by compressing a checkpoint into the
repository. Compression does not make the repository suitable for model
distribution and makes version history unnecessarily large.
