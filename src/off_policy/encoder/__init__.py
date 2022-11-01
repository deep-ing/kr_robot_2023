

def get_encoder(name, env, out_features):
    from .resnet import ResNet
    from .cnn import CNN
    encoder = {
        'cnn' :CNN,
        'resnet' : ResNet
    }[name]
    
    return encoder(env, out_features)
    