from collections import OrderedDict
TOKENIZER_MAPPING_NAMES = OrderedDict(
        [
            (
                "albert",
                (
                    "AlbertTokenizer" ,
                    "AlbertTokenizerFast" ,
                ),
            ),
            ("align", ("BertTokenizer", "BertTokenizerFast" )),
            ("aria", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("bark", ("BertTokenizer", "BertTokenizerFast" )),
            ("bart", ("BartTokenizer", "BartTokenizerFast")),
            (
                "barthez",
                (
                    "BarthezTokenizer" ,
                    "BarthezTokenizerFast" ,
                ),
            ),
            ("bartpho", ("BartphoTokenizer", None)),
            ("bert", ("BertTokenizer", "BertTokenizerFast" )),
            ("bert-generation", ("BertGenerationTokenizer" , None)),
            ("bert-japanese", ("BertJapaneseTokenizer", None)),
            ("bertweet", ("BertweetTokenizer", None)),
            (
                "big_bird",
                (
                    "BigBirdTokenizer" ,
                    "BigBirdTokenizerFast" ,
                ),
            ),
            ("bigbird_pegasus", ("PegasusTokenizer", "PegasusTokenizerFast" )),
            ("biogpt", ("BioGptTokenizer", None)),
            ("blenderbot", ("BlenderbotTokenizer", "BlenderbotTokenizerFast")),
            ("blenderbot-small", ("BlenderbotSmallTokenizer", None)),
            ("blip", ("BertTokenizer", "BertTokenizerFast" )),
            ("blip-2", ("GPT2Tokenizer", "GPT2TokenizerFast" )),
            ("bloom", (None, "BloomTokenizerFast" )),
            ("bridgetower", ("RobertaTokenizer", "RobertaTokenizerFast" )),
            ("bros", ("BertTokenizer", "BertTokenizerFast" )),
            ("byt5", ("ByT5Tokenizer", None)),
            (
                "camembert",
                (
                    "CamembertTokenizer" ,
                    "CamembertTokenizerFast" ,
                ),
            ),
            ("canine", ("CanineTokenizer", None)),
            (
                "chameleon",
                (
                    "LlamaTokenizer" ,
                    "LlamaTokenizerFast" ,
                ),
            ),
            ("chinese_clip", ("BertTokenizer", "BertTokenizerFast" )),
            (
                "clap",
                (
                    "RobertaTokenizer",
                    "RobertaTokenizerFast" ,
                ),
            ),
            (
                "clip",
                (
                    "CLIPTokenizer",
                    "CLIPTokenizerFast" ,
                ),
            ),
            (
                "clipseg",
                (
                    "CLIPTokenizer",
                    "CLIPTokenizerFast" ,
                ),
            ),
            ("clvp", ("ClvpTokenizer", None)),
            (
                "code_llama",
                (
                    "CodeLlamaTokenizer" ,
                    "CodeLlamaTokenizerFast" ,
                ),
            ),
            ("codegen", ("CodeGenTokenizer", "CodeGenTokenizerFast" )),
            ("cohere", (None, "CohereTokenizerFast" )),
            ("cohere2", (None, "CohereTokenizerFast" )),
            ("colpali", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("convbert", ("ConvBertTokenizer", "ConvBertTokenizerFast" )),
            (
                "cpm",
                (
                    "CpmTokenizer" ,
                    "CpmTokenizerFast" ,
                ),
            ),
            ("cpmant", ("CpmAntTokenizer", None)),
            ("ctrl", ("CTRLTokenizer", None)),
            ("data2vec-audio", ("Wav2Vec2CTCTokenizer", None)),
            ("data2vec-text", ("RobertaTokenizer", "RobertaTokenizerFast" )),
            ("dbrx", ("GPT2Tokenizer", "GPT2TokenizerFast" )),
            ("deberta", ("DebertaTokenizer", "DebertaTokenizerFast" )),
            (
                "deberta-v2",
                (
                    "DebertaV2Tokenizer" ,
                    "DebertaV2TokenizerFast" ,
                ),
            ),
            (
                "diffllama",
                (
                    "LlamaTokenizer" ,
                    "LlamaTokenizerFast" ,
                ),
            ),
            ("distilbert", ("DistilBertTokenizer", "DistilBertTokenizerFast" )),
            (
                "dpr",
                (
                    "DPRQuestionEncoderTokenizer",
                    "DPRQuestionEncoderTokenizerFast" ,
                ),
            ),
            ("electra", ("ElectraTokenizer", "ElectraTokenizerFast" )),
            ("emu3", ("GPT2Tokenizer", "GPT2TokenizerFast" )),
            ("ernie", ("BertTokenizer", "BertTokenizerFast" )),
            ("ernie_m", ("ErnieMTokenizer" , None)),
            ("esm", ("EsmTokenizer", None)),
            ("falcon", (None, "PreTrainedTokenizerFast" )),
            ("falcon_mamba", (None, "GPTNeoXTokenizerFast" )),
            (
                "fastspeech2_conformer",
                ("FastSpeech2ConformerTokenizer", None),
            ),
            ("flaubert", ("FlaubertTokenizer", None)),
            ("fnet", ("FNetTokenizer", "FNetTokenizerFast" )),
            ("fsmt", ("FSMTTokenizer", None)),
            ("funnel", ("FunnelTokenizer", "FunnelTokenizerFast" )),
            (
                "gemma",
                (
                    "GemmaTokenizer" ,
                    "GemmaTokenizerFast" ,
                ),
            ),
            (
                "gemma2",
                (
                    "GemmaTokenizer" ,
                    "GemmaTokenizerFast" ,
                ),
            ),
            ("git", ("BertTokenizer", "BertTokenizerFast" )),
            ("glm", (None, "PreTrainedTokenizerFast" )),
            ("gpt-sw3", ("GPTSw3Tokenizer" , None)),
            ("gpt2", ("GPT2Tokenizer", "GPT2TokenizerFast" )),
            ("gpt_bigcode", ("GPT2Tokenizer", "GPT2TokenizerFast" )),
            ("gpt_neo", ("GPT2Tokenizer", "GPT2TokenizerFast" )),
            ("gpt_neox", (None, "GPTNeoXTokenizerFast" )),
            ("gpt_neox_japanese", ("GPTNeoXJapaneseTokenizer", None)),
            ("gptj", ("GPT2Tokenizer", "GPT2TokenizerFast" )),
            ("gptsan-japanese", ("GPTSanJapaneseTokenizer", None)),
            ("grounding-dino", ("BertTokenizer", "BertTokenizerFast" )),
            ("groupvit", ("CLIPTokenizer", "CLIPTokenizerFast" )),
            ("helium", (None, "PreTrainedTokenizerFast" )),
            ("herbert", ("HerbertTokenizer", "HerbertTokenizerFast" )),
            ("hubert", ("Wav2Vec2CTCTokenizer", None)),
            ("ibert", ("RobertaTokenizer", "RobertaTokenizerFast" )),
            ("idefics", (None, "LlamaTokenizerFast" )),
            ("idefics2", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("idefics3", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("instructblip", ("GPT2Tokenizer", "GPT2TokenizerFast" )),
            ("instructblipvideo", ("GPT2Tokenizer", "GPT2TokenizerFast" )),
            (
                "jamba",
                (
                    "LlamaTokenizer" ,
                    "LlamaTokenizerFast" ,
                ),
            ),
            (
                "jetmoe",
                (
                    "LlamaTokenizer" ,
                    "LlamaTokenizerFast" ,
                ),
            ),
            ("jukebox", ("JukeboxTokenizer", None)),
            (
                "kosmos-2",
                (
                    "XLMRobertaTokenizer" ,
                    "XLMRobertaTokenizerFast" ,
                ),
            ),
            ("layoutlm", ("LayoutLMTokenizer", "LayoutLMTokenizerFast" )),
            ("layoutlmv2", ("LayoutLMv2Tokenizer", "LayoutLMv2TokenizerFast" )),
            ("layoutlmv3", ("LayoutLMv3Tokenizer", "LayoutLMv3TokenizerFast" )),
            ("layoutxlm", ("LayoutXLMTokenizer", "LayoutXLMTokenizerFast" )),
            ("led", ("LEDTokenizer", "LEDTokenizerFast" )),
            ("lilt", ("LayoutLMv3Tokenizer", "LayoutLMv3TokenizerFast" )),
            (
                "llama",
                (
                    "LlamaTokenizer" ,
                    "LlamaTokenizerFast" ,
                ),
            ),
            ("llava", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("llava_next", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("llava_next_video", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("llava_onevision", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("longformer", ("LongformerTokenizer", "LongformerTokenizerFast" )),
            (
                "longt5",
                (
                    "T5Tokenizer" ,
                    "T5TokenizerFast" ,
                ),
            ),
            ("luke", ("LukeTokenizer", None)),
            ("lxmert", ("LxmertTokenizer", "LxmertTokenizerFast" )),
            ("m2m_100", ("M2M100Tokenizer" , None)),
            ("mamba", (None, "GPTNeoXTokenizerFast" )),
            ("mamba2", (None, "GPTNeoXTokenizerFast" )),
            ("marian", ("MarianTokenizer" , None)),
            (
                "mbart",
                (
                    "MBartTokenizer" ,
                    "MBartTokenizerFast" ,
                ),
            ),
            (
                "mbart50",
                (
                    "MBart50Tokenizer" ,
                    "MBart50TokenizerFast" ,
                ),
            ),
            ("mega", ("RobertaTokenizer", "RobertaTokenizerFast" )),
            ("megatron-bert", ("BertTokenizer", "BertTokenizerFast" )),
            ("mgp-str", ("MgpstrTokenizer", None)),
            (
                "mistral",
                (
                    "LlamaTokenizer" ,
                    "LlamaTokenizerFast" ,
                ),
            ),
            (
                "mixtral",
                (
                    "LlamaTokenizer" ,
                    "LlamaTokenizerFast" ,
                ),
            ),
            ("mllama", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("mluke", ("MLukeTokenizer" , None)),
            ("mobilebert", ("MobileBertTokenizer", "MobileBertTokenizerFast" )),
            ("modernbert", (None, "PreTrainedTokenizerFast" )),
            ("moonshine", (None, "PreTrainedTokenizerFast" )),
            ("moshi", (None, "PreTrainedTokenizerFast" )),
            ("mpnet", ("MPNetTokenizer", "MPNetTokenizerFast" )),
            ("mpt", (None, "GPTNeoXTokenizerFast" )),
            ("mra", ("RobertaTokenizer", "RobertaTokenizerFast" )),
            (
                "mt5",
                (
                    "MT5Tokenizer" ,
                    "MT5TokenizerFast" ,
                ),
            ),
            ("musicgen", ("T5Tokenizer", "T5TokenizerFast" )),
            ("musicgen_melody", ("T5Tokenizer", "T5TokenizerFast" )),
            ("mvp", ("MvpTokenizer", "MvpTokenizerFast" )),
            ("myt5", ("MyT5Tokenizer", None)),
            ("nemotron", (None, "PreTrainedTokenizerFast" )),
            ("nezha", ("BertTokenizer", "BertTokenizerFast" )),
            (
                "nllb",
                (
                    "NllbTokenizer" ,
                    "NllbTokenizerFast" ,
                ),
            ),
            (
                "nllb-moe",
                (
                    "NllbTokenizer" ,
                    "NllbTokenizerFast" ,
                ),
            ),
            (
                "nystromformer",
                (
                    "AlbertTokenizer" ,
                    "AlbertTokenizerFast" ,
                ),
            ),
            ("olmo", (None, "GPTNeoXTokenizerFast" )),
            ("olmo2", (None, "GPTNeoXTokenizerFast" )),
            ("olmoe", (None, "GPTNeoXTokenizerFast" )),
            (
                "omdet-turbo",
                ("CLIPTokenizer", "CLIPTokenizerFast" ),
            ),
            ("oneformer", ("CLIPTokenizer", "CLIPTokenizerFast" )),
            (
                "openai-gpt",
                ("OpenAIGPTTokenizer", "OpenAIGPTTokenizerFast" ),
            ),
            ("opt", ("GPT2Tokenizer", "GPT2TokenizerFast" )),
            ("owlv2", ("CLIPTokenizer", "CLIPTokenizerFast" )),
            ("owlvit", ("CLIPTokenizer", "CLIPTokenizerFast" )),
            ("paligemma", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            (
                "pegasus",
                (
                    "PegasusTokenizer" ,
                    "PegasusTokenizerFast" ,
                ),
            ),
            (
                "pegasus_x",
                (
                    "PegasusTokenizer" ,
                    "PegasusTokenizerFast" ,
                ),
            ),
            (
                "perceiver",
                (
                    "PerceiverTokenizer",
                    None,
                ),
            ),
            (
                "persimmon",
                (
                    "LlamaTokenizer" ,
                    "LlamaTokenizerFast" ,
                ),
            ),
            ("phi", ("CodeGenTokenizer", "CodeGenTokenizerFast" )),
            ("phi3", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("phimoe", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("phobert", ("PhobertTokenizer", None)),
            ("pix2struct", ("T5Tokenizer", "T5TokenizerFast" )),
            ("pixtral", (None, "PreTrainedTokenizerFast" )),
            ("plbart", ("PLBartTokenizer" , None)),
            ("prophetnet", ("ProphetNetTokenizer", None)),
            ("qdqbert", ("BertTokenizer", "BertTokenizerFast" )),
            (
                "qwen2",
                (
                    "Qwen2Tokenizer",
                    "Qwen2TokenizerFast" ,
                ),
            ),
            ("qwen2_5_vl", ("Qwen2Tokenizer", "Qwen2TokenizerFast" )),
            ("qwen2_audio", ("Qwen2Tokenizer", "Qwen2TokenizerFast" )),
            (
                "qwen2_moe",
                (
                    "Qwen2Tokenizer",
                    "Qwen2TokenizerFast" ,
                ),
            ),
            ("qwen2_vl", ("Qwen2Tokenizer", "Qwen2TokenizerFast" )),
            ("rag", ("RagTokenizer", None)),
            ("realm", ("RealmTokenizer", "RealmTokenizerFast" )),
            (
                "recurrent_gemma",
                (
                    "GemmaTokenizer" ,
                    "GemmaTokenizerFast" ,
                ),
            ),
            (
                "reformer",
                (
                    "ReformerTokenizer" ,
                    "ReformerTokenizerFast" ,
                ),
            ),
            (
                "rembert",
                (
                    "RemBertTokenizer" ,
                    "RemBertTokenizerFast" ,
                ),
            ),
            ("retribert", ("RetriBertTokenizer", "RetriBertTokenizerFast" )),
            ("roberta", ("RobertaTokenizer", "RobertaTokenizerFast" )),
            (
                "roberta-prelayernorm",
                ("RobertaTokenizer", "RobertaTokenizerFast" ),
            ),
            ("roc_bert", ("RoCBertTokenizer", None)),
            ("roformer", ("RoFormerTokenizer", "RoFormerTokenizerFast" )),
            ("rwkv", (None, "GPTNeoXTokenizerFast" )),
            (
                "seamless_m4t",
                (
                    "SeamlessM4TTokenizer" ,
                    "SeamlessM4TTokenizerFast" ,
                ),
            ),
            (
                "seamless_m4t_v2",
                (
                    "SeamlessM4TTokenizer" ,
                    "SeamlessM4TTokenizerFast" ,
                ),
            ),
            ("siglip", ("SiglipTokenizer" , None)),
            ("speech_to_text", ("Speech2TextTokenizer" , None)),
            ("speech_to_text_2", ("Speech2Text2Tokenizer", None)),
            ("speecht5", ("SpeechT5Tokenizer" , None)),
            ("splinter", ("SplinterTokenizer", "SplinterTokenizerFast")),
            (
                "squeezebert",
                ("SqueezeBertTokenizer", "SqueezeBertTokenizerFast" ),
            ),
            ("stablelm", (None, "GPTNeoXTokenizerFast" )),
            ("starcoder2", ("GPT2Tokenizer", "GPT2TokenizerFast" )),
            (
                "switch_transformers",
                (
                    "T5Tokenizer" ,
                    "T5TokenizerFast" ,
                ),
            ),
            (
                "t5",
                (
                    "T5Tokenizer" ,
                    "T5TokenizerFast" ,
                ),
            ),
            ("tapas", ("TapasTokenizer", None)),
            ("tapex", ("TapexTokenizer", None)),
            ("transfo-xl", ("TransfoXLTokenizer", None)),
            ("tvp", ("BertTokenizer", "BertTokenizerFast" )),
            (
                "udop",
                (
                    "UdopTokenizer" ,
                    "UdopTokenizerFast" ,
                ),
            ),
            (
                "umt5",
                (
                    "T5Tokenizer" ,
                    "T5TokenizerFast" ,
                ),
            ),
            ("video_llava", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("vilt", ("BertTokenizer", "BertTokenizerFast" )),
            ("vipllava", ("LlamaTokenizer", "LlamaTokenizerFast" )),
            ("visual_bert", ("BertTokenizer", "BertTokenizerFast" )),
            ("vits", ("VitsTokenizer", None)),
            ("wav2vec2", ("Wav2Vec2CTCTokenizer", None)),
            ("wav2vec2-bert", ("Wav2Vec2CTCTokenizer", None)),
            ("wav2vec2-conformer", ("Wav2Vec2CTCTokenizer", None)),
            ("wav2vec2_phoneme", ("Wav2Vec2PhonemeCTCTokenizer", None)),
            ("whisper", ("WhisperTokenizer", "WhisperTokenizerFast" )),
            ("xclip", ("CLIPTokenizer", "CLIPTokenizerFast" )),
            (
                "xglm",
                (
                    "XGLMTokenizer" ,
                    "XGLMTokenizerFast" ,
                ),
            ),
            ("xlm", ("XLMTokenizer", None)),
            ("xlm-prophetnet", ("XLMProphetNetTokenizer" , None)),
            (
                "xlm-roberta",
                (
                    "XLMRobertaTokenizer" ,
                    "XLMRobertaTokenizerFast" ,
                ),
            ),
            (
                "xlm-roberta-xl",
                (
                    "XLMRobertaTokenizer" ,
                    "XLMRobertaTokenizerFast" ,
                ),
            ),
            (
                "xlnet",
                (
                    "XLNetTokenizer" ,
                    "XLNetTokenizerFast" ,
                ),
            ),
            (
                "xmod",
                (
                    "XLMRobertaTokenizer" ,
                    "XLMRobertaTokenizerFast" ,
                ),
            ),
            (
                "yoso",
                (
                    "AlbertTokenizer" ,
                    "AlbertTokenizerFast" ,
                ),
            ),
            (
                "zamba",
                (
                    "LlamaTokenizer" ,
                    "LlamaTokenizerFast" ,
                ),
            ),
            (
                "zamba2",
                (
                    "LlamaTokenizer" ,
                    "LlamaTokenizerFast" ,
                ),
            ),
        ]
    )
