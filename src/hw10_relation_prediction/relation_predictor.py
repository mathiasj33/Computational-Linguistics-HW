import sys

def main(argv):
    """ This script trains a relation prediction model, and outputs predictions for development data.
    It takes three filenames as arguments:

    python3 ./hw10_relation_predictor/relation_predictor.py <train_file> <dev_file> <prediction_for_dev_file>

    The files <train_file> and <dev_file> have the following tab-separated column format:

    "relational_subject_entity" \t "relation_name" \t "relational_object_entity" \t "sentence"

    The sentence is white-space separated, and subject / object entities are replaced by the dummy tokens <SUBJ> / <OBJ>.
    The script can assume that all relation types occur at least once in the training data.

    The file name <prediction_for_dev_file> indicates where the model prediction for the development data should be
    saved to. The output format is the predicted "relation_name" for each line in <dev_file>.

    If the output format is correct, the following will calculate the accuracy of the model:

    python3 ./hw10_relation_predictor/evaluate.py <dev_file> <prediction_for_dev_file>
    """
    train_file, dev_file, prediction_for_dev_file = argv
    pass

if __name__ == "__main__":
    main(sys.argv[1:])
