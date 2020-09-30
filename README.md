# CodeSample

This is a sample code folder.

Files:
1. EAD_model.py - The code file is implemented in Python 3.8. It is a section of Entropy-based Anomaly Detection Model for Controller Area Network bus of automotives,
designed as part of my research thesis. 

2. Output folder includes 2 pickle files, an excel sheet and a heatmap , outputted as part of the code execution

Note: 
The input file is not included in the repository due to NDA signed with the source organisation. 
The input file is a txt file with nearly 100000 rows of the format - timestamp, message. 
The message is of the format 't'+ ID(3 bytes) + Data Length Count(1byte) + payload (max 8bytes)+ ignore bits[4bytes].
