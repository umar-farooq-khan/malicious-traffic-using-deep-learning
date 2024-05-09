Abstract—The emergence of AI-based cyber attacks poses a
significant challenge to existing detection systems, necessitating
the adaptation of newer systems to detect these new threats.
Additionally, the proliferation of newer types of devices and
architectures further complicates the cybersecurity landscape.
At present, there exists a gap in the capability of detection
systems to effectively address both AI-generated attacks and
the diversity of network architectures. It is fairly impossible
to segment the network, hence to meet these challenges, we
require one versatile detection system for identifying malicious
traffic across diverse network types, newer architectures, AIgenerated threats. However, existing datasets lack representation
of multi-environment networks. To address this, we utilize the
M-En dataset which represents both; traditional IP-based and
IoT traffic. Our methodology involves applying Bi-Directional
GRU and Bi-Directional LSTM and then stacking them on top
of each other (BiGRU-BiLSTM), then further ensembling them
to improve the robustness of the model which indicated that
the best performing generated model achieved an accuracy of
0.972, a precision of 0.986 in another ensembled model while
BiLSTM gave 0.98 accuracy. All the models have advantage on
each other. These findings underscore the efficacy of ensemble
learning approaches in enhancing the detection capabilities of
multi-environment traffic detection systems.
Index Terms—Ensemble Learning, Cyber security, Cyber intrusion detection system Network security, Cyber attack detection
system, AI Generated cyber attacks, Malicious Traffic Detection
