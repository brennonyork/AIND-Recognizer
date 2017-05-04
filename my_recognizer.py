import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    sequences = test_set.get_all_sequences()

    # iterate over each sequence by word_id
    for word_id in sequences.keys():
        word_prob = dict()
        
        # iterate over each model and score
        for word, model in models.items():
            X, lengths = test_set.get_item_Xlengths(word_id)

            try:
                word_prob[word] = model.score(X, lengths)
            except:
                word_prob[word] = float("-inf")

        probabilities.insert(word_id, word_prob)
        # for guess, take the maximal value found in the probabilities, then
        # take the word for that maximal value
        guesses.insert(word_id, max(word_prob.items(), key=lambda x: x[1])[0])

    return (probabilities, guesses)
