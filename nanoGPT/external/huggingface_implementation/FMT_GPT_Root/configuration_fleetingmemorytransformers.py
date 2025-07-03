from transformers import PretrainedConfig
    
class GPTConfig(PretrainedConfig):
    model_type = "fleetingmemorytransformers"

    def __init__(
        self,
        block_size=1024,
        vocab_size=50304,  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        head_size_qkv=None,
        ffw_dim=None,
        wm_mask=False,
        wm_decay_length=1024,
        wm_decay_rate=1,
        wm_decay_type="linear",
        wm_decay_echoic_memory=1,
        wm_setting_type="old",  # old or new
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.head_size_qkv = head_size_qkv
        self.ffw_dim = ffw_dim
        self.wm_mask = wm_mask
        self.wm_decay_length = wm_decay_length
        self.wm_decay_rate = wm_decay_rate
        self.wm_decay_type = wm_decay_type
        self.wm_decay_echoic_memory = wm_decay_echoic_memory
        self.wm_setting_type = wm_setting_type


