## Text classifier

This is a text classifier created using Random Forest Classifier which detects harmful comments. The main objective of this project
was to use **MLflow** in practice.

## Dataset
The dataset was imported from https://klejbenchmark.com/. The original citation below.

*@article{ptaszynski2019results,
title={Results of the PolEval 2019 Shared Task 6: First Dataset and Open Shared Task for Automatic Cyberbullying Detection in Polish Twitter},
author={Ptaszynski, Michal and Pieciukiewicz, Agata and Dyba{\l}a, Pawe{\l}},
journal={Proceedings of the PolEval 2019 Workshop},
publisher={Institute of Computer Science, Polish Academy of Sciences},
pages={89},
year={2019}
}*

It contains comments from twitter which are classified as harmful (1) and non-harmful (0).

## MLflow
An experiment was conducted to set a probability threshold for classification. The goal was to optimize values of recall and precision.
Ten different possibilities were explored.

<img width="1892" height="858" alt="Zrzut ekranu 2025-10-02 175047" src="https://github.com/user-attachments/assets/88129e90-be3b-46ab-ba0e-99bb82f1bd9d" />

Based on the results, it can be inferred that threshold around 0.22 is the most optimal because recall and precision values are similar and quite good.
It can be assumed that we need to have both values over 0.4 and this threshold complies these requirements.
