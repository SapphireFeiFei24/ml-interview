# ML Coding Interview

## 1. Setup and Clarifying Questions (1-5min)
> Ask clarifications then restate the assumptions.
> Don't rush to coding.

### Clarifying Questions
1. Input types and shapes
2. Output format
   1. Single numbers, vector per-class, batched
3. Tie-breaking behavior
4. Labels binary or multi-class or continuous value
5. Size constraints(fits in memory?), latency vs throughput priorities
    1. Fits in memory: small-batch
6. Any expected numeric stability or weighting(sample weights)?
   1. Weight Sampling: loss or metrics is calculated based on the weights



## 2. Problem Decomposition & Design (5-10min)
> Break problem into components
> Describe 2-3 alternative approaches and pick one
> Trade-offs based on runtime, memory, simplicity

## 3. Implement using TDD/iterative Coding (10-35min)
> Write code based on tests. Fix code based on edge cases.

## 4. Scale-up and Tradeoffs (35-50 min)
> Make changes based on new constraints proposed by the interviewer
> (Scale to big data, streaming, concurrency etc)

## 5. Tests Again with Corner Cases
> Think of worse-case complexity and show how to catch regressions


## 6. Wrap Up & Ask Questions 

