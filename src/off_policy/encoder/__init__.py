

def get_encoder(name, env, out_features):
    from .resnet import ResNet
    from .cnn import CNN
    from .one_cnn import OneCNN
    encoder = {
        'one_cnn' : OneCNN,
        'cnn' :CNN,
        'resnet' : ResNet
    }[name]
    
    return encoder(env, out_features)
    