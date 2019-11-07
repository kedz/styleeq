import argparse
from pathlib import Path
import json
import plum
from plum.seq2seq.search import BeamSearch
import re
from collections import defaultdict
import textwrap
import random
from get_style_json import create_json
from subprocess import Popen


CONFIG = """

local PM = import 'PM.libsonnet';


local ds = PM.s2s.parallel_jsonl(
    "{data_dir}/SOURCE_10a.jsonl",
    "{data_dir}/TARGET_10a.jsonl",
    name="valid");

local src_tkn_vocab = PM.vocab.load("{styleeq_dir}/vocabs/source_tokens.pth");
local src_lem_vocab = PM.vocab.load("{styleeq_dir}/vocabs/source_lemmas.pth");
local pos_fine_vocab = PM.vocab.load("{styleeq_dir}/vocabs/pos_fine.pth");
local pos_coarse_vocab = PM.vocab.load("{styleeq_dir}/vocabs/pos_coarse.pth");
local ctrl_genre_vocab = PM.vocab.load("{styleeq_dir}/vocabs/ctrl_genre.pth", name="genre");

local tgt_tkn_vocab = PM.vocab.load("{styleeq_dir}/vocabs/target_tokens.pth");


local ctrl_list = ["conjunction", "determiner", "thirdNeutralPerson",
                   "thirdFemalePerson", "thirdMalePerson",
                   "firstPerson", "secondPerson", "thirdPerson",
                   "helperVerbs", "negation",
                   "simplePreposition",
                   "positionSimplePreposition",
                   "punctuation", "countParseS", "countParseSBAR",
                   "countParseADVP", "countParseFRAG"];


local pipelines = {{
    source_tokens: [
        0, "sequence", "tokens sensored",
        PM.data.pipeline.pad_list("<sos>", end=false),
        PM.data.pipeline.vocab_lookup(src_tkn_vocab),
    ],
    source_lemmas: [
        0, "sequence", "lemmas sensored",
        PM.data.pipeline.pad_list("<sos>", end=false),
        PM.data.pipeline.vocab_lookup(src_lem_vocab),
    ],
    source_pos_fine: [0, "sequence", "pos ptb",
        PM.data.pipeline.pad_list("<sos>", end=false),
        PM.data.pipeline.vocab_lookup(pos_fine_vocab),
    ],
    source_pos_coarse: [
        0, "sequence", "pos uni",
        PM.data.pipeline.pad_list("<sos>", end=false),
        PM.data.pipeline.vocab_lookup(pos_coarse_vocab),
    ],
}} + {{

    [ctrl]: [
        0, "controls", ctrl,
        {{
            __plum_type__: "dataio.pipeline.threshold_feature",
            thresholds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        }},
        PM.data.pipeline.long_tensor(),
    ],
    for ctrl in ctrl_list
}};

local collate_funcs = {{
    source_tokens: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
    source_lemmas: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
    source_pos_fine: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
    source_pos_coarse: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
}} + {{
    [ctrl]: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
    for ctrl in ctrl_list
}};


local batches = PM.data.batches(
    ds,
    batch_size=1,
    num_workers=1,
    pipelines=pipelines,
    collate_funcs=collate_funcs,
    sort=true,
    sort_key=[0, "sequence", "tokens sensored", PM.data.pipeline.len()],
    sort_descending=true,
); 

[batches, tgt_tkn_vocab, ds]
"""

def load_model(ckpt_dir):
    meta = json.loads((ckpt_dir / "ckpt.metadata.json").read_text())
    model_path = ckpt_dir / meta['optimal_checkpoint']
    return plum.load(model_path).eval()

def load_database(ds):
    database = defaultdict(lambda: defaultdict(list))
    for example in ds:
        genre = example[0]['controls']['genre']
        count = len(example[0]["sequence"]["tokens sensored"])
        database[genre][count].append(example[0])
    return database
                
def get_close_sent(base, new, database, verbose=False):
    """
    Return list of source objs most similar to base source obj w genre new.

    :param base: source json obj as string (json.loads(base))
    :param new: string of new genre, e.g. 'scifi'
    :param source: string path to file with source objs e.g. 'SOURCE_80.jsonl'
    :param verbose: boolean, if True prints some info as it runs
    :return options: list of dict
    """

    l = len(base["sequence"]["tokens sensored"])
    options = database[new][l]

    if verbose:
        print('same len', len(options))

    def slim_down_options(options, count_func, n=25, v=''):
        """Slim options if more than n left."""
        if len(options) > 100:
            options_slim = []
            c = count_func(base)
            for obj in options:
                if c == count_func(obj):
                    options_slim.append(obj)
            if len(options_slim) > n:
                options = options_slim
                if verbose:
                    print(v, len(options))
        return options
    # select ones w same number of PROPN
    def f(o):
        return o['sequence']['proper nouns'].count(' ')
    options = slim_down_options(options, f, v='same num PROPN')

    # select ones w same number of NOUNS
    def f(o):
        return o['sequence']['pos uni'].count('NOUN')
    options = slim_down_options(options, f, v='same num NOUNS')

    # select ones w same number of VERBS
    def f(o):
        return o['sequence']['pos uni'].count('VERB')
    options = slim_down_options(options, f, v='same num VERBS')

    # select ones w same number of ADJ
    def f(o):
        return o['sequence']['pos uni'].count('ADJ')
    options = slim_down_options(options, f, v='same num ADJ')

    return options

def make_transfer_inputs(orig, database, genre, num_opts=8):
    options = list(get_close_sent(orig, genre, database))
    random.shuffle(options)

    batch = []
    for opt in options[:num_opts]:
        batch.append(
            [
                {
                    "sequence": orig["sequence"], 
                    "controls": opt["controls"],
                    "source_string": " ".join(opt["sequence"]["original"]),
                }
            ]
        )

    return batch

def format_output(output_str, propn_list=None, indent="    "):
    if propn_list is None:
        propn_list = []
    else: 
        propn_list = list(propn_list)

    while propn_list and "PROPN" in output_str:
        output_str = re.sub('PROPN', propn_list.pop(0), output_str, 1)

    return textwrap.fill(
        output_str, 
        initial_indent=indent, 
        subsequent_indent=indent)


def generate_and_print(src, transfer_styles, model, batches, target_vocab,
                       database, num_opts):
    options = {
        style: make_transfer_inputs(src, database, style, num_opts)
        for style in transfer_styles
    }
    batch = [[src],] 
    for style in transfer_styles:
        batch.extend(options[style])
    batch = batches._collate_fn(batch)

    encoder_state, controls_state = model.encode(batch)
    search = BeamSearch(beam_size=8, max_steps=100, 
                        vocab=target_vocab)
    search(model.decoder, encoder_state, controls=controls_state)
    outputs = [" ".join(output) for output in search.output()]

    propn_list = list(src["sequence"]["proper nouns"])
    reconstruction = format_output(outputs[0], propn_list, indent="  ")

    print("Original:\n")
    print(textwrap.fill(src["source_string"], initial_indent="  ",
          subsequent_indent="  "))
    print()
    print("Reconstruction:\n")
    print(reconstruction)
    print()

    offset = 1
    for style in transfer_styles:
        for i in range(num_opts):
            print("Transfer -> {} ({})".format(style, i))
            print("  original:")
            print()
            print(format_output(options[style][i][0]["source_string"]))
            print()
            print("  transfer:")
            print()
            print(format_output(outputs[offset + i], propn_list)) 
            print()
        offset += num_opts
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("styleeq_dir", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("corenlp_dir", type=Path)
    parser.add_argument("--run", type=str, default="run1")
    parser.add_argument("--num-opts", type=int, default=2)
    args = parser.parse_args()

    cnlp_proc = Popen(
        ["java", "-mx4g", "-cp", "*", 
         "edu.stanford.nlp.pipeline.StanfordCoreNLPServer"], 
        cwd=str(args.corenlp_dir.resolve()))


    config = CONFIG.format(styleeq_dir=args.styleeq_dir,
                           data_dir=args.data_dir)
    plum_parser = plum.parser.PlumParser()
    (batches, target_vocab, ds), plum_pointers = plum_parser.parse_string(
        config)

    model = load_model(
        args.styleeq_dir / "train" / args.run / "model_checkpoints"
    )

    database = load_database(ds)

    # Example 1 from directly from features.
    example1 = {
        "sequence": {
            "tokens sensored": ["men", "said", "species", "most", "widely", 
                                "divergent", "PROPN", "imaginable"], 
            "lemmas sensored": ["man", "say", "species", "most", "widely", 
                                "divergent", "PROPN", "imaginable"], 
            "pos uni": ["NOUN", "VERB", "NOUN", "ADV", "ADV", "ADJ", "PROPN", 
                        "ADJ"], 
            "pos ptb": ["NNS", "VBN", "NN", "RBS", "RB", "JJ", "NNP", "JJ"], 
            "proper nouns": ["man"], 
        },    
        "controls": {
            "simplePreposition": 1, "positionSimplePreposition": 0, 
            "allPreposition": 1, "firstPerson": 0, "secondPerson": 1, 
            "thirdNeutralPerson": 0, "thirdFemalePerson": 0, 
            "thirdMalePerson": 0, "thirdPerson": 0, "anyPerson": 1, 
            "punctuation": 0, "length": 15, "determiner": 3, "conjunction": 0,
            "helperVerbs": 2, "negation": 0, "countParseS": 2, 
            "countParseSBAR": 1, "countParseADVP": 0, "countParseFRAG": 0, 
            "genre": "scifi",
        },
        "source_string": "Some men have said that your species is the most widely divergent from Man imaginable.",
    }

    generate_and_print(example1, ["gothic", "philosophy"], model, batches, 
                       target_vocab, database, args.num_opts)


    # Example 2, extracting features from string.
    example2_string = """
        In retirement she had acquired tranquillity, and had almost lost the 
        consciousness of those sorrows which yet threw a soft and not 
        unpleasing shade over her character.
    """
    example2_string = re.sub(r"(\n| )+", " ", example2_string).strip()

    example2 = json.loads(create_json(example2_string, source=True))
    example2["source_string"] = example2_string
    generate_and_print(example2, ["scifi", "philosophy"], model, batches, 
                       target_vocab, database, args.num_opts)
     

    cnlp_proc.kill()
    cnlp_proc.wait()
if __name__ == "__main__":
    main()    
