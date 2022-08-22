## Robustness_Evaluation_Machine_Learning_based_Phishing_URL_Detectors (MLPU)

This repository contains the code for Evaluating the robustness of MLPU models (manuscript submitted to TDSC). This presents our code for our evidence-based approach for systematically analysing the robustness of Machine Learning (ML)-based Phishing URL (MLPU) detection systems. A MLPU is a cybersecurity system that provides the first level of defence against phishing attacks by classifying a web URL as phishing or benign. MLPU systems are widely adopted as a part of email servers and as browser plugins to protect users from being victims of phishing attacks that are predominantly used for ransomware. 
Given the critical role of MLPU systems in cyber defence, it is important to test the robustness of such systems. Our research is aimed at devising and evaluating approaches to benchmarking the robustness of MLPU detectors. The reported work shows how to automatically generate and apply Adversarial URLs (AUs) against MLPU systems. Moreover, we assessed the adversarial URLs for their realizability, validity and deceptiveness. Subsequently, we reproduced tested 50 baseline MLPU detectors (both traditional and deep learning-based) against the generated AUs and reported their adversarial performance (robustness). Further, we have statistically examined the results, and identified several vulnerabilities in these systems. Finally, based on our results from the research reported in this paper, we have provided some evaluation challenges and provided recommendations for improving the practical adaptability of these systems.

ul>
  <li>URLBUG: Repository for Generation of Adversarial Deceptive URL using URLBUG </li>
  <li>Reproducing_Machine_Learning_based_Phishing_URL_detectors: Code to Reproduce state-of-the-art MLPU Detectors </li>
  <li></Dataset: Contains dataset to train the MLPU models and Adversarial Deceptive URL dataset to test and adversarially train MLPU Modelsli>
  <li> Defence:
     <ul>
       <li> Contains code to adversarially train the MLPU models</li>
       <li> Contains code to train Ensemble - MetaClassifier for improving MLPU Models robustness </li>
  </ul>
  </li>
  <li>AdditionalAttacks_Analysis: Code and results of applying popular NLP attacks on MLPU models</li>
</ul>
