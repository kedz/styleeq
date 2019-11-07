import json
import spacy
from pycorenlp import StanfordCoreNLP
corenlp = StanfordCoreNLP('http://localhost:9000')
nlp = spacy.load('en_core_web_sm')


def clean_word(word):
    """Return word with leading and trailing punctuation removed."""
    return word.lower().strip(',;:.-\n\'\"')

PARAMS = {
    'firstPerson': ['i', "i'd", "i'll", "i'm", 'me', 'mine', 'my', 'myself',
                    'we', "we'd", "we're", "we'll",
                    'our', 'ours'],
    'secondPerson': ['you', "you'd", "you'll", "your", "yours", "yourself"],
    'thirdNeutralPerson': ["they", "their", "they'd", "they'll", "theirs",
                           "themself", "themselves", "it"],
    'thirdFemalePerson': ["she", "her", "she'd", "she'll", "hers", "herself",
                          "she's"],
    'thirdMalePerson': ["he", "him", "he'd", "he'll", "his", "hiself", "he's"],
    'simplePreposition': ['about', 'after', 'as', 'astride', 'before',
                          'beyond', 'by', 'despite', 'during', 'except', 'for',
                          'from', 'into', 'notwithstanding', 'of', 'off', 
                          'out', 'past', 'per', 'sans', 'since', 'than',
                          'through', 'thoughout', 'till', 'to', 'unlike',
                          'until', 'upside', 'versus', 'via', 'with', 'within',
                          'without'],
    'positionSimplePreposition': ['aboard', 'above', 'across', 'against',
                                  'along', 'alongside', 'amid', 'among',
                                  'around', 'at', 'atop', 'ontop', 'behind',
                                  'below', 'beneath', 'beside', 'besides',
                                  'between', 'down', 'in', 'inside', 'near',
                                  'on', 'onto', 'opposite', 'outside', 'over',
                                  'toward', 'towards', 'under', 'underneath',
                                  'up', 'upon'],
    'punctuation': [',', ';', '-', '_', '(', ':'],
    'determiner': ['a', 'an', 'the', 'this', 'that', 'some'],
    'conjunction': ['and', 'or', 'nor', 'but', 'so', 'yet'],
    'helperVerbs': ['be', 'is', 'am', 'are', 'was', 'were', 'can', 'could', 
                    'dare', 'do', 'have', 'has', 'may', 'might', 'must', 
                    'should', 'will', 'would', 'had'],
    'negation': ['not', "n't"]
    }

PARAMS['thirdPerson'] = PARAMS['thirdNeutralPerson'] + PARAMS['thirdFemalePerson'] + PARAMS['thirdMalePerson']





def count_personal_pronouns(sentence, params):
    """Return dict of {'firstPerson':count, 'secondPerson': count, etc.}

    :param sentence: lower case string
    :param params: dict containing match strings
    :return data: dict
    """
    data = {'firstPerson': 0, 'secondPerson': 0,
            'thirdNeutralPerson': 0, 'thirdFemalePerson': 0,
            'thirdMalePerson': 0, 'thirdPerson': 0}
    for item in sentence.split(' '):
        word = clean_word(item)
        for key in data:
            for pronoun in params[key]:
                if word == pronoun:
                    data[key] += 1
    data['anyPerson'] = data['firstPerson'] + data['secondPerson'] + data['thirdNeutralPerson'] + data['thirdFemalePerson'] + data['thirdMalePerson']
    return data


def count_prepositions(sentence, params):
    """Return dict of {'simplePreposition': count,
                       'positionSimplePreposition': count,
                       'allPreposition': count}

    lists of preps are from wikipedia

    :param sentence: lower case string
    :param params: dict containing match strings
    :return data: dict
    """
    sentence = ' ' + sentence
    data = {'simplePreposition': 0, 'positionSimplePreposition': 0}
    for key in ['simplePreposition', 'positionSimplePreposition']:
        for prep in params[key]:
            prep = ' ' + prep
            data[key] += sentence.count(prep)
    data['allPreposition'] = data['simplePreposition'] + data['positionSimplePreposition']
    return data


def count_punctuation(sentence, params):
    """Return dict of {'punctuation': count}. Doesn't count end punctuation.

    :param sentence: string
    :param params: dict containing match strings
    :return data: dict
    """
    data = {'punctuation': 0}
    for p in params['punctuation']:
        data['punctuation'] += sentence.count(p)
    return data


def count_determiner(sentence, params):
    """Return dict of {'determiner': count}.

    :param sentence: string
    :param params: dict containing match strings
    :return data: dict
    """
    data = {'determiner': 0}
    for item in sentence.split(' '):
        word = clean_word(item)
        for d in params['determiner']:
            if word == d:
                data['determiner'] += 1
    return data


def count_conjunction(sentence, params):
    """Return dict of {'conjunction': count}.

    :param sentence: string
    :param params: dict containing match strings
    :return data: dict
    """
    data = {'conjunction': 0}
    for item in sentence.split(' '):
        word = clean_word(item)
        for d in params['conjunction']:
            if word == d:
                data['conjunction'] += 1
    return data


def count_helper_verbs(sentence, params):
    """Return dict of {'helperVerbs': count}.

    :param sentence: string
    :param params: dict containing match strings
    :return data: dict
    """
    data = {'helperVerbs': 0}
    for item in sentence.split(' '):
        word = clean_word(item)
        for d in params['helperVerbs']:
            if word == d:
                data['helperVerbs'] += 1
    return data


def count_negation(sentence, params):
    """Return dict of {'negation': count}.

    :param sentence: string
    :param params: dict containing match strings
    :return data: dict
    """
    data = {'negation': 0}
    for item in sentence.split(' '):
        word = clean_word(item)
        if word == 'not':
            data['negation'] += 1
        elif "n't" in word:
            data['negation'] += 1
    return data


def count_length(sentence, params={}):
    """Return dict of {'length': integer}, just using space counts."""
    return {'length': len(sentence.split(' '))}


def prep_sent(sentence, params):
    """Return sentence as list of tokens without punctuation, determiners, 
    pronouns, simple prepositions or conjunctions.

    Also return list of lemmas of these tokens.
    Also return list of POS (universal and penn tree bank).

    :param sentence: spacy doc
    :param params: dict containing match strings
    :return data: list of str
    """
    keys = ['firstPerson', 'secondPerson', 'thirdPerson', 
            'punctuation', 
            'simplePreposition',
            'positionSimplePreposition',
            'conjunction', 
            'determiner',
            'helperVerbs',
            'negation']
    match_list = []
    for k in keys:
        match_list += params[k]
    
    pos_list = ['PUNCT', 'SYM']
    
    data = {
        'original': [],
        'original sensored': [],
        'tokens': [],
        'lemmas': [],
        'tokens sensored': [],
        'lemmas sensored': [],
        'pos uni': [],
        'pos ptb': [],
        'proper nouns': []
    }
    
    for token in sentence:
        if token.pos_ == 'SPACE':
            continue
        
        data['original'].append(token.text.lower())
        if token.pos_ == 'PROPN':
            data['original sensored'].append('PROPN')
        else:
            data['original sensored'].append(token.text.lower())
        
        word = clean_word(token.text)
        if word not in match_list and token.pos_ not in pos_list:
            data['tokens'].append(word)
            data['lemmas'].append(token.lemma_)
            data['pos uni'].append(token.pos_)
            data['pos ptb'].append(token.tag_)
            if token.pos_ == 'PROPN':
                data['tokens sensored'].append('PROPN')
                data['lemmas sensored'].append('PROPN')
                data['proper nouns'].append(word)
            else:
                data['tokens sensored'].append(word)
                data['lemmas sensored'].append(token.lemma_)
    return data


def count_parse_feats(sentence, param={}, pprint=False):
    """Return dict of count of S, SBAR, ADVP in constituency parse."""
    output = corenlp.annotate(sentence, properties={'annotators': 'parse', 'outputFormat': 'json'})
    parse = output['sentences'][0]['parse']
    data = {}
    data['countParseS'] = parse.count('S\n')
    data['countParseSBAR'] = parse.count('SBAR')
    data['countParseADVP'] = parse.count('ADVP')
    data['countParseFRAG'] = parse.count('FRAG')
    
    if pprint:
        print(parse)

    return data

FUNCTIONLIST = [count_prepositions, count_personal_pronouns, count_punctuation, 
                count_length, count_determiner, count_conjunction, 
                count_helper_verbs, count_negation, count_parse_feats]


def create_json(sentence, source=False, metadata={}):
    """Return a json obj based on features in sentence.

    Each function in functionlist should take a sentence and params 
    and return a dict of features and results.

    Assumes document is in gutenberg style, with line breaks between
    sentences sometimes; uses double linebreak as sign of paragraphs.

    Skips any sentence with < 5 or > 25 content tokens.

    {"reference_string": "heard the gunshot the silencer.",
     "sequence": {
         "original": ["i", "heard", "the", ...],  # as it would be in target
         "tokens": ["heard", "the", ...],   # as it would be in source
         "lemmas": ["hear", "the", ...],
         "pos uni": [],
         "pos ptb": [],
         "tokens sensored": [ner tokens replaced with PROPN],
         "lemmas sensored": [ditto],
         "proper nouns": [ Dracula, Califoria]
         },
     "controls": {"count_1p": 5, "count_3p": 2, .... },
    }

    :param sentence: string to analyze
    :param source: if true, return source version, else return target version
    :param metadata: extra data for controls
    :return: json
    """
    sent = nlp(sentence)
    # don't include sentences that have too few content words
    sent_data = prep_sent(sent, PARAMS)
    if len(sent_data['tokens']) < 5 or len(sent_data['tokens']) > 25:
        raise ValueError('input sentence is too short or too long.')

    data = {}
    if source:  # going to be source
        data['reference_string'] = ' '.join(sent_data['tokens'])
        data['sequence'] = sent_data
    else:  # going to be target
        data['reference_string'] = sent.text
        data['sequence'] = {}
        data['sequence']['tokens'] = sent_data['original']
        data['sequence']['tokens sensored'] = sent_data['original sensored']
    data['controls'] = {}

    for function in FUNCTIONLIST:
        sent_lower = sent.text.lower()
        data['controls'].update(function(sent_lower, PARAMS))
    data['controls'].update(metadata)
    return json.dumps(data)


if __name__ == "__main__":
    while True:
        text = input("Enter a sentence: ")
        source = ''
        while source not in ['s', 't']:
            source = input("Do you want the source [s] or target [t]? ")
            if source not in ['s', 't']:
                print("Please type 's' or 't'.")
        src = source == 's'
        obj = create_json(text, source=src)
        print(obj)
