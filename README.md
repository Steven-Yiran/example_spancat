# Steps to Run

1. Install required dependencies in `requirements.txt`.
2. Run the `span_categorizer_training` notebook to train span classification models with cross validation and use them to produce predicted probabilities and other necessary data.
3. Run the `find_label_errors_in_span_classification_dataset` notebook to detect label issues with cleanlab. This notebook needs to be run in an environment where you have the cleanlab version that has the `cleanlab.experimental.span_classification` module.
