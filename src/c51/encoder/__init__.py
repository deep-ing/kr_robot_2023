

def get_encoder(name, env, out_features):
    from .resnet import ResNet
    encoder = {
        'resnet' : ResNet
    }[name]
    
    return encoder(env, out_features)
    