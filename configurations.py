import all_constants as ac


def base():
    config = {}

    config['embed_dim'] = 512
    config['ff_dim'] = 2048
    config['num_enc_layers'] = 6
    config['num_dec_layers'] = 6

    # architecture
    config['use_bias'] = True
    config['fix_norm'] = True
    config['mask_logit'] = True

    # data
    config['regen_batches'] = False
    config['data_dup'] = False # if True, concate consecutive examples to generate new examples
    config['data_dup_rand'] = False # if True, shuffle training data before concat sentences
    config['length_limit'] = 1024

    config['clip_grad'] = 1.0
    config['lr_scheduler'] = ac.ORG_WU
    config['warmup_steps'] = 8000
    config['lr'] = 3e-4
    config['lr_scale'] = 1.
    config['lr_decay'] = 0.8
    config['stop_lr'] = 5e-5
    config['patience'] = 3
    config['alpha'] = 0.7
    config['label_smoothing'] = 0.1
    config['batch_size'] = 4096
    config['epoch_size'] = 1000
    config['max_epochs'] = 200
    config['dropout'] = 0.3
    config['att_dropout'] = 0.3
    config['ff_dropout'] = 0.3
    config['word_dropout'] = 0.1

    # Decoding
    config['beam_size'] = 4
    config['beam_alpha'] = 0.6

    return config


def en2vi_rnn():
    config = base()
    config['epoch_size'] = 1500
    config['num_enc_layers'] = 2
    config['num_dec_layers'] = 2
    config['att_dropout'] = 0.
    config['dropout'] = 0.2
    return config


def ted_gl2en_rnn():
    config = base()
    config['num_enc_layers'] = 2
    config['num_dec_layers'] = 2

    config['att_dropout'] = 0.
    config['dropout'] = 0.3
    config['word_dropout'] = 0. # for gl2en, no word-dropout works better

    config['max_epochs'] = 1000
    config['epoch_size'] = 100
    return config


def ted_sk2en_rnn():
    config = base()
    config['num_enc_layers'] = 2
    config['num_dec_layers'] = 2

    config['att_dropout'] = 0.
    config['dropout'] = 0.2

    config['max_epochs'] = 200
    config['epoch_size'] = 600
    return config
