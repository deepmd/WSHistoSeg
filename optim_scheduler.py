from torch.optim import SGD, Adam, lr_scheduler


def _get_model_params_for_opt(cfg, model):
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['encoder.features.'],  # features
        'resnet': ['encoder.layer4.', 'classification_head.', 'seg_head.', 'proj_head.'],  # CLASSIFIER
        'inception': ['encoder.Mixed', 'encoder.Conv2d_1',
                      'encoder.Conv2d_2',
                      'encoder.Conv2d_3', 'encoder.Conv2d_4'],  # features
    }

    def param_features_substring_list(arch):
        for key in _FEATURE_PARAM_LAYER_PATTERNS:
            if arch.startswith(key):
                return _FEATURE_PARAM_LAYER_PATTERNS[key]
        raise KeyError("Fail to recognize the architecture {}".format(arch))

    def string_contains_any(string, substring_list):
        for substring in substring_list:
            if substring in string:
                return True
        return False

    param_features = []
    param_heads = []
    for name, parameter in model.named_parameters():
        if string_contains_any(
                name,
                param_features_substring_list(cfg.encoder_name)):
            if cfg.encoder_name in ('vgg16', ''):
                param_features.append(parameter)
            elif cfg.encoder_name == 'resnet50':
                param_heads.append(parameter)
        else:
            if cfg.encoder_name in ('vgg16', 'inceptionv3'):
                param_heads.append(parameter)
            elif cfg.encoder_name == 'resnet50':
                param_features.append(parameter)

    return [
        {'params': param_features, 'lr': cfg.learning_rate},
        {'params': param_heads, 'lr': cfg.learning_rate * cfg.lr_heads_ratio}
    ]


def get_optimizer(parameters, opt):
    if opt.optimizer == "SGD":
        return SGD(
            params=parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
            nesterov=True
        )
    elif opt.optimizer == "Adam":
        return Adam(
            params=parameters,
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
        )
    else:
        raise ValueError(f"Specified optimizer name '{opt.optimizer}' is not valid.")


def get_optim_scheduler(model, opt):
    params_group = _get_model_params_for_opt(opt, model)
    optimizer = get_optimizer(params_group, opt)

    lambda_poly = lambda iters: pow((1.0 - iters / opt.max_iters), opt.power)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)

    return optimizer, scheduler