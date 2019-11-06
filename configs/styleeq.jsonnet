local PM = import 'PM.libsonnet';
 
local root_dir = "./";

local train_ds = PM.s2s.parallel_jsonl(
    root_dir + "/literary_style_data/SOURCE_80.jsonl",
    root_dir + "/literary_style_data/TARGET_80.jsonl",
    name="train", mmap=true);

local valid_ds = PM.s2s.parallel_jsonl(
    root_dir + "/literary_style_data/SOURCE_10a.3k.jsonl",
    root_dir + "/literary_style_data/TARGET_10a.3k.jsonl",
    name="valid", mmap=true);

local test_ds = PM.s2s.parallel_jsonl(
    root_dir + "/literary_style_data/SOURCE_10b.jsonl",
    root_dir + "/literary_style_data/TARGET_10b.jsonl",
    name="test", mmap=true);

local src_tkn_vocab = PM.vocab.new(
    "source_tokens",
    train_ds,
    [0, "sequence", "tokens sensored"],
    start_token="<sos>",
    stop_token="<eos>",
    pad_token="<pad>",
    unknown_token="<unk>",
    at_least=5,
    top_k=50000,
);

local tgt_tkn_vocab = PM.vocab.new(
    "target_tokens",
    train_ds,
    [1, "sequence", "tokens sensored"],
    start_token="<sos>",
    stop_token="<eos>",
    pad_token="<pad>",
    unknown_token="<unk>",
    at_least=5,
    top_k=50000,
);

local src_lem_vocab = PM.vocab.new(
    "source_lemmas",
    train_ds,
    [0, "sequence", "lemmas sensored"],
    start_token="<sos>",
    stop_token="<eos>",
    pad_token="<pad>",
    unknown_token="<unk>",
    at_least=5,
    top_k=50000,
);

local pos_fine_vocab = PM.vocab.new(
    "pos_fine",
    train_ds,
    [0, "sequence", "pos ptb"],
    start_token="<sos>",
    stop_token="<eos>",
    pad_token="<pad>",
);

local pos_coarse_vocab = PM.vocab.new(
    "pos_coarse",
    train_ds,
    [0, "sequence", "pos uni"],
    start_token="<sos>",
    stop_token="<eos>",
    pad_token="<pad>",
);

local ctrl_list = ["conjunction", "determiner", "thirdNeutralPerson", 
                   "thirdFemalePerson", "thirdMalePerson",
                   "firstPerson", "secondPerson", "thirdPerson",
                   "helperVerbs", "negation",
                   "simplePreposition", 
                   "positionSimplePreposition",
                   "punctuation", "countParseS", "countParseSBAR",
                   "countParseADVP", "countParseFRAG"];

local pipelines = {
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
    target_inputs: [
        1, "sequence", "tokens sensored",
        PM.data.pipeline.pad_list("<sos>", end=false),
        PM.data.pipeline.vocab_lookup(tgt_tkn_vocab),
    ],
    target_outputs: [
        1, "sequence", "tokens sensored",
        PM.data.pipeline.pad_list("<eos>", start=false),
        PM.data.pipeline.vocab_lookup(tgt_tkn_vocab),
    ],
    input_references: [0, "reference_string"],
    target_references: [1, "reference_string"],
} + {

    [ctrl]: [
        0, "controls", ctrl,
        {
            __plum_type__: "dataio.pipeline.threshold_feature",
            thresholds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        },
        PM.data.pipeline.long_tensor(),
    ],
    for ctrl in ctrl_list

};

local collate_funcs = {
    source_tokens: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
    source_lemmas: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
    source_pos_fine: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
    source_pos_coarse: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
    target_inputs: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
    target_outputs: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
} + {
    [ctrl]: PM.data.pipeline.batch_sequence_ndtensor(0, 0, 1),
    for ctrl in ctrl_list};


local train_batches = PM.data.batches(
    train_ds,
    batch_size=64,
    num_workers=4,
    pipelines=pipelines,
    collate_funcs=collate_funcs,
    sort=true,
    sort_key=[0, "sequence", "tokens sensored", PM.data.pipeline.len()],
    sort_descending=true,
);

local valid_batches = PM.data.batches(
    valid_ds,
    batch_size=16,
    num_workers=4,
    pipelines=pipelines,
    collate_funcs=collate_funcs,
    sort=true,
    sort_key=[0, "sequence", "tokens sensored", PM.data.pipeline.len()],
    sort_descending=true,
);

local test_batches = PM.data.batches(
    test_ds,
    batch_size=8,
    num_workers=4,
    pipelines=pipelines,
    collate_funcs=collate_funcs,
    sort=true,
    sort_key=[0, "sequence", "tokens sensored", PM.data.pipeline.len()],
    sort_descending=true,
);

local loss_function = PM.loss.cross_entropy(
    labels_field="target_outputs",
    padding_index=PM.vocab.pad_index(tgt_tkn_vocab)
);

local searches = {
    greedy: {
        __plum_type__: "seq2seq.search.greedy",
        max_steps: 75,
        vocab: tgt_tkn_vocab,
    },
    beam8: {
        __plum_type__: "seq2seq.search.beam",
        max_steps: 75,
        beam_size: 8,
        vocab: tgt_tkn_vocab,
    },
};


local controls = {
    [ctrl]: {
        __plum_type__: "layers.embedding",
        in_feats: std.length(pipelines[ctrl][3].thresholds) + 1,
        out_feats: 50,
        dropout: 0.25,
    }
    for ctrl in ctrl_list
};


local model = PM.s2s.models.rnn(
    512, 
    {
        source_tokens: src_tkn_vocab, 
        source_lemmas: src_lem_vocab,
        source_pos_fine: pos_fine_vocab, 
        source_pos_coarse: pos_coarse_vocab
    }, 
    tgt_tkn_vocab,
    encoder_inputs=["source_tokens", "source_lemmas", "source_pos_fine",
                    "source_pos_coarse"],
    emb_sizes={source_tokens: 128, source_lemmas: 128, 
               source_pos_fine: 64, source_pos_coarse: 32,
               target_inputs: 512},
    controls=controls,
);

local metrics = PM.metrics.dict({
    greedy: PM.s2s.metrics.eval_script(
        root_dir + "eval_scripts/eval.py", 
        ["search", "greedy"], 
        "target_references"),
    beam8: PM.s2s.metrics.eval_script(
        root_dir + "eval_scripts/eval.py", 
        ["search", "beam8"], 
        "target_references"),
});

local loggers = {
    greedy: {
        __plum_type__: "loggers.search_output_logger",
        file_prefix: "greedy.checkpoint",
        search_fields: ["search", "greedy"],
        input_fields: "input_references",
        reference_fields: "target_references",
    },
    beam8: {
        __plum_type__: "loggers.search_output_logger",
        file_prefix: "beam8.checkpoint",
        search_fields: ["search", "beam8"],
        input_fields: "input_references",
        reference_fields: "target_references",
    },
};

local checkpoints = {
    __plum_type__: "checkpoints.topk",
    k: 3,
    criterion: ["valid", "metrics", "beam8", "BLEU"],
    min_criterion: false,
};

[
    PM.trainer(
        model, train_batches, valid_batches,
        PM.optim.sgd(lr=0.25, weight_decay=0.0001),
        loss_function, 
        max_epochs=200,
        searches=searches,
        valid_metrics=metrics,
        valid_loggers=loggers,
        checkpoints=checkpoints,
    ),

    PM.s2s.evaluator(
        test_batches,
        searches=searches,
        loss_function=loss_function,
        metrics=metrics,
        loggers=loggers,
        name="eval-test",
    ),
]
