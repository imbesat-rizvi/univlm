from transformers import models
from collections import OrderedDict
import importlib
_LazyAutoMapping = models.auto.auto_factory._LazyAutoMapping
Base = models.auto.auto_factory._BaseAutoModelClass

CONFIG_MAPPING_NAMES = OrderedDict(
    [
        # Add configs here
        ("albert", "AlbertConfig"),
        ("align", "AlignConfig"),
        ("altclip", "AltCLIPConfig"),
        ("aria", "AriaConfig"),
        ("aria_text", "AriaTextConfig"),
        ("audio-spectrogram-transformer", "ASTConfig"),
        ("autoformer", "AutoformerConfig"),
        ("bamba", "BambaConfig"),
        ("bark", "BarkConfig"),
        ("bart", "BartConfig"),
        ("beit", "BeitConfig"),
        ("bert", "BertConfig"),
        ("bert-generation", "BertGenerationConfig"),
        ("big_bird", "BigBirdConfig"),
        ("bigbird_pegasus", "BigBirdPegasusConfig"),
        ("biogpt", "BioGptConfig"),
        ("bit", "BitConfig"),
        ("blenderbot", "BlenderbotConfig"),
        ("blenderbot-small", "BlenderbotSmallConfig"),
        ("blip", "BlipConfig"),
        ("blip-2", "Blip2Config"),
        ("bloom", "BloomConfig"),
        ("bridgetower", "BridgeTowerConfig"),
        ("bros", "BrosConfig"),
        ("camembert", "CamembertConfig"),
        ("canine", "CanineConfig"),
        ("chameleon", "ChameleonConfig"),
        ("chinese_clip", "ChineseCLIPConfig"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionConfig"),
        ("clap", "ClapConfig"),
        ("clip", "CLIPConfig"),
        ("clip_text_model", "CLIPTextConfig"),
        ("clip_vision_model", "CLIPVisionConfig"),
        ("clipseg", "CLIPSegConfig"),
        ("clvp", "ClvpConfig"),
        ("code_llama", "LlamaConfig"),
        ("codegen", "CodeGenConfig"),
        ("cohere", "CohereConfig"),
        ("cohere2", "Cohere2Config"),
        ("colpali", "ColPaliConfig"),
        ("conditional_detr", "ConditionalDetrConfig"),
        ("convbert", "ConvBertConfig"),
        ("convnext", "ConvNextConfig"),
        ("convnextv2", "ConvNextV2Config"),
        ("cpmant", "CpmAntConfig"),
        ("ctrl", "CTRLConfig"),
        ("cvt", "CvtConfig"),
        ("dac", "DacConfig"),
        ("data2vec-audio", "Data2VecAudioConfig"),
        ("data2vec-text", "Data2VecTextConfig"),
        ("data2vec-vision", "Data2VecVisionConfig"),
        ("dbrx", "DbrxConfig"),
        ("deberta", "DebertaConfig"),
        ("deberta-v2", "DebertaV2Config"),
        ("decision_transformer", "DecisionTransformerConfig"),
        ("deformable_detr", "DeformableDetrConfig"),
        ("deit", "DeiTConfig"),
        ("depth_anything", "DepthAnythingConfig"),
        ("deta", "DetaConfig"),
        ("detr", "DetrConfig"),
        ("diffllama", "DiffLlamaConfig"),
        ("dinat", "DinatConfig"),
        ("dinov2", "Dinov2Config"),
        ("dinov2_with_registers", "Dinov2WithRegistersConfig"),
        ("distilbert", "DistilBertConfig"),
        ("donut-swin", "DonutSwinConfig"),
        ("dpr", "DPRConfig"),
        ("dpt", "DPTConfig"),
        ("efficientformer", "EfficientFormerConfig"),
        ("efficientnet", "EfficientNetConfig"),
        ("electra", "ElectraConfig"),
        ("emu3", "Emu3Config"),
        ("encodec", "EncodecConfig"),
        ("encoder-decoder", "EncoderDecoderConfig"),
        ("ernie", "ErnieConfig"),
        ("ernie_m", "ErnieMConfig"),
        ("esm", "EsmConfig"),
        ("falcon", "FalconConfig"),
        ("falcon_mamba", "FalconMambaConfig"),
        ("fastspeech2_conformer", "FastSpeech2ConformerConfig"),
        ("flaubert", "FlaubertConfig"),
        ("flava", "FlavaConfig"),
        ("fnet", "FNetConfig"),
        ("focalnet", "FocalNetConfig"),
        ("fsmt", "FSMTConfig"),
        ("funnel", "FunnelConfig"),
        ("fuyu", "FuyuConfig"),
        ("gemma", "GemmaConfig"),
        ("gemma2", "Gemma2Config"),
        ("git", "GitConfig"),
        ("glm", "GlmConfig"),
        ("glpn", "GLPNConfig"),
        ("gpt-sw3", "GPT2Config"),
        ("gpt2", "GPT2Config"),
        ("gpt_bigcode", "GPTBigCodeConfig"),
        ("gpt_neo", "GPTNeoConfig"),
        ("gpt_neox", "GPTNeoXConfig"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseConfig"),
        ("gptj", "GPTJConfig"),
        ("gptsan-japanese", "GPTSanJapaneseConfig"),
        ("granite", "GraniteConfig"),
        ("granitemoe", "GraniteMoeConfig"),
        ("graphormer", "GraphormerConfig"),
        ("grounding-dino", "GroundingDinoConfig"),
        ("groupvit", "GroupViTConfig"),
        ("helium", "HeliumConfig"),
        ("hiera", "HieraConfig"),
        ("hubert", "HubertConfig"),
        ("ibert", "IBertConfig"),
        ("idefics", "IdeficsConfig"),
        ("idefics2", "Idefics2Config"),
        ("idefics3", "Idefics3Config"),
        ("idefics3_vision", "Idefics3VisionConfig"),
        ("ijepa", "IJepaConfig"),
        ("imagegpt", "ImageGPTConfig"),
        ("informer", "InformerConfig"),
        ("instructblip", "InstructBlipConfig"),
        ("instructblipvideo", "InstructBlipVideoConfig"),
        ("jamba", "JambaConfig"),
        ("jetmoe", "JetMoeConfig"),
        ("jukebox", "JukeboxConfig"),
        ("kosmos-2", "Kosmos2Config"),
        ("layoutlm", "LayoutLMConfig"),
        ("layoutlmv2", "LayoutLMv2Config"),
        ("layoutlmv3", "LayoutLMv3Config"),
        ("led", "LEDConfig"),
        ("levit", "LevitConfig"),
        ("lilt", "LiltConfig"),
        ("llama", "LlamaConfig"),
        ("llava", "LlavaConfig"),
        ("llava_next", "LlavaNextConfig"),
        ("llava_next_video", "LlavaNextVideoConfig"),
        ("llava_onevision", "LlavaOnevisionConfig"),
        ("longformer", "LongformerConfig"),
        ("longt5", "LongT5Config"),
        ("luke", "LukeConfig"),
        ("lxmert", "LxmertConfig"),
        ("m2m_100", "M2M100Config"),
        ("mamba", "MambaConfig"),
        ("mamba2", "Mamba2Config"),
        ("marian", "MarianConfig"),
        ("markuplm", "MarkupLMConfig"),
        ("mask2former", "Mask2FormerConfig"),
        ("maskformer", "MaskFormerConfig"),
        ("maskformer-swin", "MaskFormerSwinConfig"),
        ("mbart", "MBartConfig"),
        ("mctct", "MCTCTConfig"),
        ("mega", "MegaConfig"),
        ("megatron-bert", "MegatronBertConfig"),
        ("mgp-str", "MgpstrConfig"),
        ("mimi", "MimiConfig"),
        ("mistral", "MistralConfig"),
        ("mixtral", "MixtralConfig"),
        ("mllama", "MllamaConfig"),
        ("mobilebert", "MobileBertConfig"),
        ("mobilenet_v1", "MobileNetV1Config"),
        ("mobilenet_v2", "MobileNetV2Config"),
        ("mobilevit", "MobileViTConfig"),
        ("mobilevitv2", "MobileViTV2Config"),
        ("modernbert", "ModernBertConfig"),
        ("moonshine", "MoonshineConfig"),
        ("moshi", "MoshiConfig"),
        ("mpnet", "MPNetConfig"),
        ("mpt", "MptConfig"),
        ("mra", "MraConfig"),
        ("mt5", "MT5Config"),
        ("musicgen", "MusicgenConfig"),
        ("musicgen_melody", "MusicgenMelodyConfig"),
        ("mvp", "MvpConfig"),
        ("nat", "NatConfig"),
        ("nemotron", "NemotronConfig"),
        ("nezha", "NezhaConfig"),
        ("nllb-moe", "NllbMoeConfig"),
        ("nougat", "VisionEncoderDecoderConfig"),
        ("nystromformer", "NystromformerConfig"),
        ("olmo", "OlmoConfig"),
        ("olmo2", "Olmo2Config"),
        ("olmoe", "OlmoeConfig"),
        ("omdet-turbo", "OmDetTurboConfig"),
        ("oneformer", "OneFormerConfig"),
        ("open-llama", "OpenLlamaConfig"),
        ("openai-gpt", "OpenAIGPTConfig"),
        ("opt", "OPTConfig"),
        ("owlv2", "Owlv2Config"),
        ("owlvit", "OwlViTConfig"),
        ("paligemma", "PaliGemmaConfig"),
        ("patchtsmixer", "PatchTSMixerConfig"),
        ("patchtst", "PatchTSTConfig"),
        ("pegasus", "PegasusConfig"),
        ("pegasus_x", "PegasusXConfig"),
        ("perceiver", "PerceiverConfig"),
        ("persimmon", "PersimmonConfig"),
        ("phi", "PhiConfig"),
        ("phi3", "Phi3Config"),
        ("phimoe", "PhimoeConfig"),
        ("pix2struct", "Pix2StructConfig"),
        ("pixtral", "PixtralVisionConfig"),
        ("plbart", "PLBartConfig"),
        ("poolformer", "PoolFormerConfig"),
        ("pop2piano", "Pop2PianoConfig"),
        ("prophetnet", "ProphetNetConfig"),
        ("pvt", "PvtConfig"),
        ("pvt_v2", "PvtV2Config"),
        ("qdqbert", "QDQBertConfig"),
        ("qwen2", "Qwen2Config"),
        ("qwen2_audio", "Qwen2AudioConfig"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoderConfig"),
        ("qwen2_moe", "Qwen2MoeConfig"),
        ("qwen2_vl", "Qwen2VLConfig"),
        ("rag", "RagConfig"),
        ("realm", "RealmConfig"),
        ("recurrent_gemma", "RecurrentGemmaConfig"),
        ("reformer", "ReformerConfig"),
        ("regnet", "RegNetConfig"),
        ("rembert", "RemBertConfig"),
        ("resnet", "ResNetConfig"),
        ("retribert", "RetriBertConfig"),
        ("roberta", "RobertaConfig"),
        ("roberta-prelayernorm", "RobertaPreLayerNormConfig"),
        ("roc_bert", "RoCBertConfig"),
        ("roformer", "RoFormerConfig"),
        ("rt_detr", "RTDetrConfig"),
        ("rt_detr_resnet", "RTDetrResNetConfig"),
        ("rwkv", "RwkvConfig"),
        ("sam", "SamConfig"),
        ("seamless_m4t", "SeamlessM4TConfig"),
        ("seamless_m4t_v2", "SeamlessM4Tv2Config"),
        ("segformer", "SegformerConfig"),
        ("seggpt", "SegGptConfig"),
        ("sew", "SEWConfig"),
        ("sew-d", "SEWDConfig"),
        ("siglip", "SiglipConfig"),
        ("siglip_vision_model", "SiglipVisionConfig"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderConfig"),
        ("speech_to_text", "Speech2TextConfig"),
        ("speech_to_text_2", "Speech2Text2Config"),
        ("speecht5", "SpeechT5Config"),
        ("splinter", "SplinterConfig"),
        ("squeezebert", "SqueezeBertConfig"),
        ("stablelm", "StableLmConfig"),
        ("starcoder2", "Starcoder2Config"),
        ("superpoint", "SuperPointConfig"),
        ("swiftformer", "SwiftFormerConfig"),
        ("swin", "SwinConfig"),
        ("swin2sr", "Swin2SRConfig"),
        ("swinv2", "Swinv2Config"),
        ("switch_transformers", "SwitchTransformersConfig"),
        ("t5", "T5Config"),
        ("table-transformer", "TableTransformerConfig"),
        ("tapas", "TapasConfig"),
        ("textnet", "TextNetConfig"),
        ("time_series_transformer", "TimeSeriesTransformerConfig"),
        ("timesformer", "TimesformerConfig"),
        ("timm_backbone", "TimmBackboneConfig"),
        ("timm_wrapper", "TimmWrapperConfig"),
        ("trajectory_transformer", "TrajectoryTransformerConfig"),
        ("transfo-xl", "TransfoXLConfig"),
        ("trocr", "TrOCRConfig"),
        ("tvlt", "TvltConfig"),
        ("tvp", "TvpConfig"),
        ("udop", "UdopConfig"),
        ("umt5", "UMT5Config"),
        ("unispeech", "UniSpeechConfig"),
        ("unispeech-sat", "UniSpeechSatConfig"),
        ("univnet", "UnivNetConfig"),
        ("upernet", "UperNetConfig"),
        ("van", "VanConfig"),
        ("video_llava", "VideoLlavaConfig"),
        ("videomae", "VideoMAEConfig"),
        ("vilt", "ViltConfig"),
        ("vipllava", "VipLlavaConfig"),
        ("vision-encoder-decoder", "VisionEncoderDecoderConfig"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderConfig"),
        ("visual_bert", "VisualBertConfig"),
        ("vit", "ViTConfig"),
        ("vit_hybrid", "ViTHybridConfig"),
        ("vit_mae", "ViTMAEConfig"),
        ("vit_msn", "ViTMSNConfig"),
        ("vitdet", "VitDetConfig"),
        ("vitmatte", "VitMatteConfig"),
        ("vitpose", "VitPoseConfig"),
        ("vitpose_backbone", "VitPoseBackboneConfig"),
        ("vits", "VitsConfig"),
        ("vivit", "VivitConfig"),
        ("wav2vec2", "Wav2Vec2Config"),
        ("wav2vec2-bert", "Wav2Vec2BertConfig"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerConfig"),
        ("wavlm", "WavLMConfig"),
        ("whisper", "WhisperConfig"),
        ("xclip", "XCLIPConfig"),
        ("xglm", "XGLMConfig"),
        ("xlm", "XLMConfig"),
        ("xlm-prophetnet", "XLMProphetNetConfig"),
        ("xlm-roberta", "XLMRobertaConfig"),
        ("xlm-roberta-xl", "XLMRobertaXLConfig"),
        ("xlnet", "XLNetConfig"),
        ("xmod", "XmodConfig"),
        ("yolos", "YolosConfig"),
        ("yoso", "YosoConfig"),
        ("zamba", "ZambaConfig"),
        ("zoedepth", "ZoeDepthConfig"),
    ]
)
MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "BlipForConditionalGeneration"),
        # ("blip-2", "Blip2ForConditionalGeneration"),
        ("chameleon", "ChameleonForConditionalGeneration"),
        ("git", "GitForCausalLM"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("instructblip", "InstructBlipForConditionalGeneration"),
        ("instructblipvideo", "InstructBlipVideoForConditionalGeneration"),
        # ("kosmos-2", "Kosmos2ForConditionalGeneration"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("mllama", "MllamaForConditionalGeneration"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("pix2struct", "Pix2StructForConditionalGeneration"),
        ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
        ("video_llava", "VideoLlavaForConditionalGeneration"),
        ("vipllava", "VipLlavaForConditionalGeneration"),
        # ("vision-encoder-decoder", "VisionEncoderDecoderModel"),
    ]
)
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        # ("bamba", "BambaForCausalLM"),
        ("bart", "BartForCausalLM"),
        ("bert", "BertLMHeadModel"),
        # ("bert-generation", "BertGenerationDecoder"),
        ("big_bird", "BigBirdForCausalLM"),
        ("bigbird_pegasus", "BigBirdPegasusForCausalLM"),
        ("biogpt", "BioGptForCausalLM"),
        ("blenderbot", "BlenderbotForCausalLM"),
        # ("blenderbot-small", "BlenderbotSmallForCausalLM"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        # ("code_llama", "LlamaForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("cohere", "CohereForCausalLM"),
        # ("cohere2", "Cohere2ForCausalLM"),
        ("cpmant", "CpmAntForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        # ("data2vec-text", "Data2VecTextForCausalLM"),
        ("dbrx", "DbrxForCausalLM"),
        # ("diffllama", "DiffLlamaForCausalLM"),
        ("electra", "ElectraForCausalLM"),
        # ("emu3", "Emu3ForCausalLM"),
        ("ernie", "ErnieForCausalLM"),
        ("falcon", "FalconForCausalLM"),
        ("falcon_mamba", "FalconMambaForCausalLM"),
        ("fuyu", "FuyuForCausalLM"),
        ("gemma", "GemmaForCausalLM"),
        ("gemma2", "Gemma2ForCausalLM"),
        ("git", "GitForCausalLM"),
        ("glm", "GlmForCausalLM"),
        # ("gpt-sw3", "GPT2LMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("granite", "GraniteForCausalLM"),
        ("granitemoe", "GraniteMoeForCausalLM"),
        # ("helium", "HeliumForCausalLM"),
        ("jamba", "JambaForCausalLM"),
        ("jetmoe", "JetMoeForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("mamba", "MambaForCausalLM"),
        ("mamba2", "Mamba2ForCausalLM"),
        ("marian", "MarianForCausalLM"),
        ("mbart", "MBartForCausalLM"),
        # ("mega", "MegaForCausalLM"),
        # ("megatron-bert", "MegatronBertForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
        ("mllama", "MllamaForCausalLM"),
        ("moshi", "MoshiForCausalLM"),
        ("mpt", "MptForCausalLM"),
        ("musicgen", "MusicgenForCausalLM"),
        ("musicgen_melody", "MusicgenMelodyForCausalLM"),
        ("mvp", "MvpForCausalLM"),
        ("nemotron", "NemotronForCausalLM"),
        ("olmo", "OlmoForCausalLM"),
        # ("olmo2", "Olmo2ForCausalLM"),
        ("olmoe", "OlmoeForCausalLM"),
        # ("open-llama", "OpenLlamaForCausalLM"),
        # ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("opt", "OPTForCausalLM"),
        ("pegasus", "PegasusForCausalLM"),
        ("persimmon", "PersimmonForCausalLM"),
        ("phi", "PhiForCausalLM"),
        ("phi3", "Phi3ForCausalLM"),
        ("phimoe", "PhimoeForCausalLM"),
        ("plbart", "PLBartForCausalLM"),
        ("prophetnet", "ProphetNetForCausalLM"),
        # ("qdqbert", "QDQBertLMHeadModel"),
        ("qwen2_moe", "Qwen2MoeForCausalLM"),
        ("recurrent_gemma", "RecurrentGemmaForCausalLM"),
        ("reformer", "ReformerModelWithLMHead"),
        ("rembert", "RemBertForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        # ("roberta-prelayernorm", "RobertaPreLayerNormForCausalLM"),
        ("roc_bert", "RoCBertForCausalLM"),
        ("roformer", "RoFormerForCausalLM"),
        ("rwkv", "RwkvForCausalLM"),
        # ("speech_to_text_2", "Speech2Text2ForCausalLM"),
        ("stablelm", "StableLmForCausalLM"),
        ("starcoder2", "Starcoder2ForCausalLM"),
        # ("transfo-xl", "TransfoXLLMHeadModel"),
        ("trocr", "TrOCRForCausalLM"),
        ("whisper", "WhisperForCausalLM"),
        ("xglm", "XGLMForCausalLM"),
        ("xlm", "XLMWithLMHeadModel"),
        # ("xlm-prophetnet", "XLMProphetNetForCausalLM"),
        # ("xlm-roberta", "XLMRobertaForCausalLM"),
        # ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xmod", "XmodForCausalLM"),
        ("zamba", "ZambaForCausalLM"),
        ("qwen2", "Qwen2ForCausalLM"),
    ]
)
MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for depth estimation mapping
        ("depth_anything", "DepthAnythingForDepthEstimation"),
        ("dpt", "DPTForDepthEstimation"),
        ("glpn", "GLPNForDepthEstimation"),
        ("zoedepth", "ZoeDepthForDepthEstimation"),
    ]
)
MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict(
    [
        ("sam", "SamModel"),
    ]
)



Token = None

class Loader_Lazy(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type, attr):
        """Override to use direct imports instead of relative ones"""
        module_name = f"transformers.models.{model_type}"
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(module_name)
        return getattr(self._modules[module_name], attr)


def updater(Loader):
    models.auto.auto_factory.auto_class_update(Loader)


class appledepth:
    def _init_(self):
        self.model = None
        self.transform = None
        self.image = None
        self.f_px = None

    @staticmethod
    def env_setup():
        pass

    def load_model(self):
        from src import depth_pro
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()

    def processor(self, image_path):
        from src import depth_pro
        self.image, _, self.f_px = depth_pro.load_rgb(image_path)
        self.image = self.transform(self.image)

    def infer(self):
        prediction = self.model.infer(self.image, f_px=self.f_px)
        return prediction


# class appledepth():
#     def __init__(self):
#         self.model = None
#         self.transform = None
#         self.image = None
#         self.f_px = None

#     def env_setup(self):
#         pass

#     def load_model(self):
#         from src import depth_pro
#         self.model, self.transform = depth_pro.create_model_and_transforms()
#         self.model.eval()
        
    
#     def processor(self, image_path):
#         from src import depth_pro
#         self.image, _, self.f_px = depth_pro.load_rgb(image_path)
#         self.image = self.transform(self.image)
#         #return self.image, self.f_px

#     def infer(self):
#         prediction = self.model.infer(self.image, f_px=self.f_px)
#         return prediction

# class appledepth:
#     def _init_():
#         model = None
#         transform = None
#         image = None
#         f_px = None
    
#     @staticmethod
#     def env_setup():
#         pass
    
#     @staticmethod
#     def load_model():
#         from src import depth_pro
#         model, transform = depth_pro.create_model_and_transforms()
#         model.eval()
#         return model, transform
    
#     @staticmethod
#     def processor(image_path, transform):
#         from src import depth_pro
#         image, _, f_px = depth_pro.load_rgb(image_path)
#         image = transform(image)
#         return image, f_px
    
#     @staticmethod
#     def infer(model, image, f_px):
#         prediction = model.infer(image, f_px=f_px)
#         return prediction

Mapper = {
         "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, 
         "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES": MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
         "MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES": MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES,
         "MODEL_FOR_MASK_GENERATION_MAPPING_NAMES": MODEL_FOR_MASK_GENERATION_MAPPING_NAMES,
         "appleDepthPro": appledepth}


# flow is we add models which we want to support in the jason.
# then we add them to config_mapping_names line 7
# then you add them to the mapper line 552.
# 