from importlib import import_module


def factory(subdir, module_name, func):
    module = import_module(
        '.' + module_name, package=subdir
    )
    model = getattr(module, func)
    return model


def count_parameter(model):
    clf_param_num = sum(p.numel() for p in model.parameters())
    return clf_param_num
