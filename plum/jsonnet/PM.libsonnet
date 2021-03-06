{
    data: import 'PB_DATA.libsonnet',
    vocab: import 'PB_VOCAB.libsonnet',
    trainer: import 'PB_TRAINER.libsonnet',
    models: import 'PB_MODEL.libsonnet',
    activation: import 'activations.libsonnet',
    loss: import 'plum.loss.libsonnet',
    optim: import 'plum.optim.libsonnet',
    metrics: import 'PB_METRICS.libsonnet',
    init: import 'PB_INIT.libsonnet',
    eval: import 'PB_EVAL.libsonnet',
    s2s: import 's2s.libsonnet', 
    s2c: import 's2c.libsonnet',
    tasks: import 'plum.tasks.libsonnet',
}
