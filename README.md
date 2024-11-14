# CACScoring
Coronary Artery Calcium Scoring from Non-Contrast Cardiac CT Using Segmentation and Deep Learning with nnUNet

## Project Overview
This project focuses on developing a deep learning model for coronary artery calcium (CAC) scoring using non-contrast cardiac CT scans. The aim was to create an automated system capable of segmenting and quantifying calcifications in the coronary arteries, providing a reliable and efficient method for assessing cardiovascular risk.

### Methodology
- **Model Used**: The project utilized the **nnUNet** framework, known for its robust and adaptive nature in medical image segmentation.
- **Data Preprocessing**: An annotated dataset was processed to ensure high-quality and consistent input data for model training. This included tasks such as organizing, formatting, and filtering out anomalies.
- **Training and Optimization**: The model was trained on a GPU cluster, with performance optimization and experiments to manage long training times effectively.
- **Evaluation**: The trained model was evaluated for accuracy in segmenting and scoring CAC, with results analyzed to ensure it met clinical standards for assessing cardiovascular risks.

### Internship Report
For a detailed explanation of the project, its methodology, results, and insights, please refer to the [-->Internship report<--](Internship-Report.pdf).

### Database
For this project, I worked with the publicly available **Stanford COCA dataset**, which includes non-contrast chest CT scans used to develop machine learning models for CAC scoring.

[-->Access the Stanford COCA Dataset<--](https://aimi.stanford.edu/datasets/coca-coronary-calcium-chest-ct)


