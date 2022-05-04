# DRL backdoor attack
Pytorch implementation of "A Temporal-pattern Backdoor Attack to Deep Reinforcement Learning''.

## Abstraction
Deep reinforcement learning (DRL) has achieved significant achievements in many real-world applications. But these real-world cases typically can only provide partial observations to DRL for making decisions. However, partial state observability can be used to hide malicious behaviors for backdoors. In this paper, we explore the sequential nature of DRL and propose a novel temporal-pattern backdoor attack to DRL, whose trigger is a set of temporal constraints on a sequence of observations rather than a single observation, and the effect can be kept in a controllable duration rather than in the instant. We validate our proposed backdoor attack to a typical job scheduling task in cloud computing.  Numerous experimental results show that our backdoor can achieve excellent effectiveness, stealthiness, and sustainability. Our backdoor's average clean data accuracy and attack success rate can reach 97.8% and 97.5%, respectively.


## Requirements 

Python 3.6
PyTorch
numpy
