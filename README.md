# Disclaimer

This is an independent robustness evaluation of Improving Alignment and Robustness with Circuit Breakers. For more information about the work see the repository of the [authors](https://github.com/GraySwanAI/circuit-breakers).
We thank the authors for providing the models and for their support.

# Changes

Currently, we only adapted the softopt embedding attack used in the original paper. 
1) We use signed gradient descent instead of gradient descent as an optimizer
2) We do not initialize the soft tokens with the embeddings of a sequence of "x" tokens. Instead, we use a semantically meaningful string that was randomly chosen and not optimized.
3) We generate multiple responses for every attack and evaluate all of them for success using the judge model

# Results

The improved embedding attack achieves a 100% attack success rate (ASR) on both Mistral-7B-Instruct-v2 + RR and Llama-3-8B-Instruct + RR, improving the ASR by more than 80% compared to the original evaluation. 

## Citation
If you find this useful in your research, please consider citing our [work](https://arxiv.org/abs/2407.15902):
```
@article{schwinn2024revisiting,
  title={Revisiting the Robust Alignment of Circuit Breakers},
  author={Schwinn, Leo and Geisler, Simon},
  journal={arXiv preprint arXiv:2407.15902},
  year={2024}
}
```

and the original paper [paper](https://arxiv.org/abs/2406.04313):
```
@misc{zou2024circuitbreaker,
title={Improving Alignment and Robustness with Circuit Breakers},
author={Andy Zou and Long Phan and Justin Wang and Derek Duenas and Maxwell Lin and Maksym Andriushchenko and Rowan Wang and Zico Kolter and Matt Fredrikson and Dan Hendrycks},
year={2024},
eprint={2406.04313},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```
