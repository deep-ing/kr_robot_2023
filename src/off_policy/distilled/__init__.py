# getting the diltillation wrapper model 


def get_distilled(name, network, flags):
    from .simple import Simple 
    distilled = {
        'simple': Simple 
    }[name]
    return distilled(network, flags)