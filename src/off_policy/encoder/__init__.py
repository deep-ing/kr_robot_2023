

def get_encoder(name, env, out_features):
    from .resnet import ResNet
    from .cnn import CNN
    from .one_cnn import OneCNN
    from .mlp import MLP_DEEP, MLP_SIM
    encoder = {
        'one_cnn' : OneCNN,
        'cnn' :CNN,
        'mlp_simple' :MLP_SIM,
        'mlp_deep' :MLP_DEEP,
        'resnet' : ResNet
    }[name]
    
    return encoder(env, out_features)
    