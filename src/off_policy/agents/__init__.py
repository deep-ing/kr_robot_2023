


def get_agent(name, encoder_t, encoder_t_target, flags):
    from .c51 import C51
    from .dqn import DQN 
    model = {
        'c51' : C51, 
        'dqn' : DQN,
    }[name]
    return model(encoder_t, encoder_t_target, flags)
    