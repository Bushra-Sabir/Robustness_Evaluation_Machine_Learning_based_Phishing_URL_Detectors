## Robustness_Evaluation_Machine_Learning_based_Phishing_URL_Detectors (MLPU)

This repository contains the code for Evaluating the robustness of MLPU models (manuscript submitted to TDSC). This presents our code for our evidence-based approach for systematically analysing the robustness of Machine Learning (ML)-based Phishing URL (MLPU) detection systems. A MLPU is a cybersecurity system that provides the first level of defence against phishing attacks by classifying a web URL as phishing or benign. MLPU systems are widely adopted as a part of email servers and as browser plugins to protect users from being victims of phishing attacks that are predominantly used for ransomware. 
Given the critical role of MLPU systems in cyber defence, it is important to test the robustness of such systems. Our research is aimed at devising and evaluating approaches to benchmarking the robustness of MLPU detectors. The reported work shows how to automatically generate and apply Adversarial URLs (AUs) against MLPU systems. Moreover, we assessed the adversarial URLs for their realizability, validity and deceptiveness. Subsequently, we reproduced tested 50 baseline MLPU detectors (both traditional and deep learning-based) against the generated AUs and reported their adversarial performance (robustness). Further, we have statistically examined the results, and identified several vulnerabilities in these systems. Finally, based on our results from the research reported in this paper, we have provided some evaluation challenges and provided recommendations for improving the practical adaptability of these systems.

<ol>
  <li>URLBUG: Repository for Generation of Adversarial Deceptive URL using URLBUG </li>
  <li>Reproducing_Machine_Learning_based_Phishing_URL_detectors: Code to Reproduce state-of-the-art MLPU Detectors </li>
  <li>Dataset: Contains dataset to train the MLPU models and Adversarial Deceptive URL dataset to test and adversarially train MLPU Models</li>
  <li> Defence:
     <ul>
       <li> Contains code to adversarially train the MLPU models</li>
       <li> Contains code to train Ensemble - MetaClassifier for improving MLPU Models robustness </li>
  </ul>
  </li>
  <li>AdditionalAttacks_Analysis: Code and results of applying popular NLP attacks on MLPU models</li>
</ol>



<h1> Citation </h1>
Title: Reliability and Robustness analysis of Machine Learning based Phishing URL Detectors
Venue: <b> 2022 Transactions of Dependable and Secure Computing, Special Issue: Reliability and Robustness in AI-Based Cybersecurity Solutions </b>
ArXiv Link: https://arxiv.org/abs/2005.08454
**Authors**: Bushra Sabir, M. Ali Babar, Raj Gaire and Alsharif Abuadbba  
<p>
**Abstract**: ML-based Phishing URL (MLPU) detectors serve as the first level of defence to protect users and organisations from being victims of phishing attacks. Lately, few studies have launched successful adversarial attacks against specific MLPU detectors raising questions about their practical reliability and usage. Nevertheless, the robustness of these systems has not been extensively investigated. Therefore, the security vulnerabilities of these systems, in general, remain primarily unknown which calls for testing the robustness of these systems. In this article, we have proposed a methodology to investigate the reliability and robustness of 50 representative state-of-the-art MLPU models. Firstly, we have proposed a cost-effective Adversarial URL generator URLBUG that created an Adversarial URL dataset. Subsequently, we reproduced 50 MLPU (traditional ML and Deep learning) systems and recorded their baseline performance. Lastly, we tested the considered MLPU systems on Adversarial Dataset and analyzed their robustness and reliability using box plots and heat maps. Our results showed that the generated adversarial URLs have valid syntax and can be registered at a median annual price of $11.99. Out of 13\% of the already registered adversarial URLs, 63.94\% were used for malicious purposes. Moreover, the considered MLPU models Matthew Correlation Coefficient (MCC) dropped from a median 0.92 to 0.02 when tested against Advdata, indicating that the baseline MLPU models are unreliable in their current form. Further, our findings identified several security vulnerabilities of these systems and provided future directions for researchers to design dependable and secure MLPU systems.
</p>
<hr/>
<b> If you use URLBUG for your research, please cite </b>

  ```diff
  @article{DBLP:journals/corr/abs-2005-08454, 
  author    = {Bushra Sabir and in green
               Muhammad Ali Babar and
               Raj Gaire},
  title     = {An Evasion Attack against ML-based Phishing {URL} Detectors},
  journal   = {CoRR},
  volume    = {abs/2005.08454},
  year      = {2020},
  url       = {https://arxiv.org/abs/2005.08454},
  eprinttype = {arXiv},
  eprint    = {2005.08454},
  timestamp = {Mon, 03 Aug 2020 19:23:49 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2005-08454.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}

```
  
