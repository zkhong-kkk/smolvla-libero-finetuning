# Experiment Protocol

## Objective

Measure how continued SmolVLA fine-tuning changes closed-loop manipulation
success in the LIBERO simulator while keeping the evaluation task, initial
states, camera mapping, action horizon, and random seed fixed.

## Confirmed Task-0 Comparison

All three policies were evaluated on:

- Suite: `libero_spatial`
- Task ID: `0`
- Episodes: `10`
- Seed: `42`
- Batch size: `1`
- Action steps: `10`
- Control mode: relative (LIBERO default)

| Policy | Success rate | Successes | 95% Wilson CI |
| --- | ---: | ---: | ---: |
| Released SmolVLA LIBERO checkpoint | 100% | 10/10 | 72.2%--100% |
| Stage 1 fine-tuned checkpoint | 20% | 2/10 | 5.7%--51.0% |
| Stage 2 fine-tuned checkpoint | 60% | 6/10 | 31.3%--83.2% |

## Interpretation

The Stage 2 checkpoint improves the observed success rate by 40 percentage
points over Stage 1. This is a useful engineering signal, but the confidence
intervals overlap because only 10 episodes were evaluated. The result should be
described as a preliminary single-task improvement until the full suite is
complete.

## Full-Suite Evaluation

The Stage 2 checkpoint has been evaluated on all 10 tasks in
`libero_spatial`, with 10 episodes per task. It succeeded in 65 of 100 episodes,
for a 65.0% success rate and a 95% Wilson interval of 55.3%--73.6%.

| Policy | Scope | Success rate | Successes | 95% Wilson CI |
| --- | --- | ---: | ---: | ---: |
| Stage 2 fine-tuned checkpoint | 10 tasks, 10 episodes each | 65.0% | 65/100 | 55.3%--73.6% |
| Released SmolVLA LIBERO checkpoint | 10 tasks, 10 episodes each | 80.0% | 80/100 | 71.1%--86.7% |

The released checkpoint is 15 percentage points higher under the same
100-episode protocol. The task-0 20%/60%/100% comparison is retained to study
the effect of training stage, while the 65%/80% comparison is the primary
full-suite result.

The overall Stage 2 values are recorded in `results/spatial_suite_summary.csv`.
The final archive should also include:

- overall success rate;
- number of successes and episodes;
- Wilson confidence interval;
- per-task success rates;
- evaluation duration;
- exact checkpoint path and training step;
- LeRobot commit and Python environment.

## Recommended Claims

Safe wording for the current result:

> Fine-tuned SmolVLA on a single RTX 4090 and built a reproducible LIBERO
> evaluation pipeline. Continued fine-tuning improved success on a held-fixed
> LIBERO-Spatial task from 20% to 60% over 10 simulator episodes, and the final
> checkpoint achieved 65% success across 100 episodes covering all 10
> LIBERO-Spatial tasks.

Do not claim state-of-the-art performance or real-robot generalization from the
current experiment.
