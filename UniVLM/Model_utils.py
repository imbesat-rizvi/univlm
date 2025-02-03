
from .Package_Backbone.modeling_auto_Dicts import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES,MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,MODEL_FOR_MASKED_LM_MAPPING_NAMES,MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES,MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES, MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES,MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,MODEL_FOR_CTC_MAPPING_NAMES,MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES,MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES,MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES,MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES,MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES,MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES,MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,MODEL_FOR_MASK_GENERATION_MAPPING_NAMES, MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES, MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForImageClassification, AutoModelForMaskGeneration, AutoModelForDepthEstimation ,AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForTableQuestionAnswering,AutoModelForVisualQuestionAnswering, AutoModelForDocumentQuestionAnswering,AutoModelForTokenClassification, AutoModelForMultipleChoice, AutoModelForNextSentencePrediction, AutoModelForImageClassification, AutoModelForAudioClassification, AutoModelForCTC, AutoModelForAudioFrameClassification, AutoModelForAudioXVector, AutoModelForTextToSpectrogram, AutoModelForTextToWaveform, AutoModelForZeroShotImageClassification, AutoModelForVisualQuestionAnswering, AutoModelForImageSegmentation, AutoModelForInstanceSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation, AutoModelForVideoClassification, AutoModelForVision2Seq, AutoModelForImageTextToText, AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection, AutoModelForSpeechSeq2Seq, AutoModelForMaskedImageModeling
from .Package_Backbone.processing_auto_Dicts import PROCESSOR_MAPPING_NAMES
from .Package_Backbone.tokenization_auto_Dicts import TOKENIZER_MAPPING_NAMES
from .Package_Backbone.feature_extraction_auto_Dicts import FEATURE_EXTRACTOR_MAPPING_NAMES
from .Package_Backbone.image_processing_auto_Dicts import IMAGE_PROCESSOR_MAPPING_NAMES
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, AutoProcessor
import concurrent.futures
import re
from fuzzywuzzy import fuzz  

class HFModelSearcher:
    def __init__(self):
        self.ordered_dicts_mapping =  {
                                        "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
                                        "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
                                        "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES": MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
                                        "MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES": MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES,
                                        "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES": MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
                                        "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES": MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
                                        "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES": MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
                                        "MODEL_FOR_MASKED_LM_MAPPING_NAMES": MODEL_FOR_MASKED_LM_MAPPING_NAMES,
                                        "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES": MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES,
                                        "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES": MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES,
                                        "MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES": MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES,
                                        "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES": MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
                                        "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES": MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
                                        "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
                                        "MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES": MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
                                        "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES": MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
                                        "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES": MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES,
                                        "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
                                        "MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES": MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
                                        "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES": MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
                                        "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES": MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
                                        "MODEL_FOR_CTC_MAPPING_NAMES": MODEL_FOR_CTC_MAPPING_NAMES,
                                        "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES": MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES,
                                        "MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES": MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES,
                                        "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES": MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES,
                                        "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES": MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES,
                                        "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES": MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
                                        "MODEL_FOR_MASK_GENERATION_MAPPING_NAMES": MODEL_FOR_MASK_GENERATION_MAPPING_NAMES,
                                        "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES": MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES,
                                      }
        self.model_classes_mapping =  {
                                        "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES": AutoModelForCausalLM,
                                        "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES": AutoModelForImageClassification,
                                        "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES": AutoModelForSemanticSegmentation,
                                        "MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES": AutoModelForUniversalSegmentation,
                                        "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES": AutoModelForVideoClassification,
                                        "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES": AutoModelForVision2Seq,
                                        "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES": AutoModelForImageTextToText,
                                        "MODEL_FOR_MASKED_LM_MAPPING_NAMES": AutoModelForMaskedLM,
                                        "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES": AutoModelForObjectDetection,
                                        "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES": AutoModelForZeroShotObjectDetection,
                                        "MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES": AutoModelForDepthEstimation,
                                        "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES": AutoModelForSeq2SeqLM,
                                        "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES": AutoModelForSpeechSeq2Seq,
                                        "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES": AutoModelForSequenceClassification,
                                        "MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES": AutoModelForQuestionAnswering,
                                        "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES": AutoModelForDocumentQuestionAnswering,
                                        "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES": AutoModelForTableQuestionAnswering,
                                        "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES": AutoModelForTokenClassification,
                                        "MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES": AutoModelForMultipleChoice,
                                        "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES": AutoModelForNextSentencePrediction,
                                        "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES": AutoModelForAudioClassification,
                                        "MODEL_FOR_CTC_MAPPING_NAMES": AutoModelForCTC,
                                        "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES": AutoModelForAudioFrameClassification,
                                        "MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES": AutoModelForAudioXVector,
                                        "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES": AutoModelForTextToSpectrogram,
                                        "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES": AutoModelForTextToWaveform,
                                        "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES": AutoModelForZeroShotImageClassification,
                                        "MODEL_FOR_MASK_GENERATION_MAPPING_NAMES": AutoModelForMaskGeneration,
                                        "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES": AutoModelForVisualQuestionAnswering,
        }

    @staticmethod
    def extract_model_family(hf_path: str) -> str:
        """Extracts the model family name from a HuggingFace model path.

        This function parses a HuggingFace model path and extracts the model family
        name by splitting the path and analyzing its components.

        Args:
            hf_path (str): HuggingFace model path in format 'org/model-name' or
                'org/model-family-version'.

        Returns:
            str: The extracted model family name.

        Raises:
            ValueError: If the path format is invalid or cannot be parsed.

        Examples:
            >>> extract_model_family("facebook/opt-350m")
            'opt'
            >>> extract_model_family("microsoft/phi-2")
            'phi'
        """
        model_name = hf_path.split("/")[-1]  # Get the last part of the path (e.g., 'bert-base-uncased')

        # Remove common size-related suffixes (e.g., 'base', 'large', 'uncased', etc.)
        model_name = re.sub(r"-(base|large|small|medium|tiny|xxl|xl|uncased|cased|distill)(?=[-_]|$)", "", model_name, flags=re.IGNORECASE)

        # Remove version identifiers like numbers and letters after model names (e.g., 'v1', 'v2')
        model_name = re.sub(r"-\d+[MBGT]?(?=[-_]|$)", "", model_name, flags=re.IGNORECASE)

        # Remove specific model family suffixes (e.g., 'roberta', 'gpt', 't5', etc.)
        model_name = re.sub(r"-(roberta|gpt|t5|bart|mistral|gemma|bloom|bigbird)(?=[-_]|$)", "", model_name, flags=re.IGNORECASE)

        # Handle special cases like 'wav2vec2' and 'bert'
        if "wav2vec2" in model_name:
            model_name = re.sub(r"-CV\d+-\w+$", "", model_name, flags=re.IGNORECASE)
        elif "bert" in model_name:
            # Ensure 'bert' is preserved as the core model family
            model_name = "bert"

        # Clean up any remaining special characters
        model_name = re.sub(r"[-_]+$", "", model_name, flags=re.IGNORECASE)
        return model_name
    
    @staticmethod
    def search_in_ordered_dict(mapping_name: str, ordered_dict, config_name: str):
        """
        Searches for the config_name in a single ordered dictionary.

        Args:
            mapping_name (str): The name of the mapping to be returned if a match is found.
            ordered_dict (OrderedDict): The ordered dictionary to search within.
            config_name (str): The configuration name to search for in the ordered dictionary.

        Returns:
            tuple: A tuple containing the mapping_name and the model_family if a match is found.
            None: If no match is found.
        """
        for model_family, config in ordered_dict.items():
            if config == config_name:
                return mapping_name, model_family
        return None


    def search(self, query: str = None,config = None):
        if config is not None:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_mapping = {executor.submit(self.search_in_ordered_dict, name, od, config): name for name, od in self.ordered_dicts_mapping.items()}
                
                for future in concurrent.futures.as_completed(future_to_mapping):
                    result = future.result()
                    if result:
                        print(result)
                        return result  # Return as soon as we find the match

            return None
        else:
            query = self.extract_model_family(query)
            results = []

            # ================== Parallel Exact Matching ==================
            def exact_match_worker(dict_item):
                """Worker function for exact matching"""
                dict_name, ordered_dict = dict_item
                return [[dict_name, ordered_dict[key]] for key in ordered_dict if key == query]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit all dictionaries for parallel processing
                exact_futures = [executor.submit(exact_match_worker, item) 
                                for item in self.ordered_dicts_mapping.items()]
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(exact_futures):
                    if matches := future.result():
                        results.extend(matches)

            # ================== Parallel Fuzzy Matching ==================
            if not results:
                all_entries = []
                for dict_name, ordered_dict in self.ordered_dicts_mapping.items():
                    all_entries.extend([(dict_name, key, val) for key, val in ordered_dict.items()])

                norm_query = re.sub(r"[-_]", "", query).lower()
                
                def fuzzy_score_worker(entry):
                    """Worker function for fuzzy scoring"""
                    dict_name, key, val = entry
                    norm_key = re.sub(r"[-_]", "", key).lower()
                    return (dict_name, val, fuzz.ratio(norm_query, norm_key))

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Process all entries in parallel
                    scored_entries = list(executor.map(fuzzy_score_worker, all_entries))                
                    # Find maximum score
                    max_score = max((entry[2] for entry in scored_entries), default=0)
                    
                    # Collect all matches with max score
                    if max_score > 65:  # Only consider good matches (threshold adjustable)
                        results.extend(
                            [[entry[0], entry[1]]
                            for entry in scored_entries if entry[2] == max_score]
                        )
            return results if results else None

class HFProcessorSearcher:
    def __init__(self):
        self.ordered_dicts_mapping = {
            "PROCESSOR_MAPPING_NAMES": PROCESSOR_MAPPING_NAMES,
            "TOKENIZER_MAPPING_NAMES": TOKENIZER_MAPPING_NAMES,
            "IMAGE_PROCESSOR_MAPPING_NAMES": IMAGE_PROCESSOR_MAPPING_NAMES,
            "FEATURE_EXTRACTOR_MAPPING_NAMES": FEATURE_EXTRACTOR_MAPPING_NAMES
        }
        self.model_classes_mapping = {
            "PROCESSOR_MAPPING_NAMES": AutoProcessor,
            "TOKENIZER_MAPPING_NAMES": AutoTokenizer,
            "IMAGE_PROCESSOR_MAPPING_NAMES": AutoImageProcessor,
            "FEATURE_EXTRACTOR_MAPPING_NAMES": AutoFeatureExtractor
        }

    @staticmethod
    def extract_model_family(hf_path: str) -> str:
        """
        Extracts the core model family name from a given Hugging Face model path.
        This function processes the model path to remove size/version suffixes 
        (e.g., '-base', '-v2') and isolates the core family name (e.g., 'vit' 
        from 'vit-patch16'). The resulting family name is normalized to lowercase.
        Args:
            hf_path (str): The file path to the Hugging Face model.
        Returns:
            str: The core model family name in lowercase.
        """
        model_name = hf_path.split("/")[-1]
        
        # Remove size/version suffixes (e.g., '-base', '-v2')
        model_name = re.sub(
            r"-(base|large|small|tiny|medium|xxl|xl|\d+[mbgt]|v\d+)(?=[-_]|$)", 
            "", 
            model_name, 
            flags=re.IGNORECASE
        )
        
        # Split to isolate core family name (e.g., 'vit' from 'vit-patch16')
        core_family = model_name.split("-")[0].split("_")[0]
        return core_family.lower()  # Normalize to lowercase

    def search(self, query: str, feature_extractor=False, image_processor=False, tokenizer=False):
        """Searches for matching model processors based on query and processor type flags.

        This method performs a two-phase search:
        1. Targeted search based on specified processor type flags
        2. Fallback search across all processor types if no matches found

        Args:
            query (str): Model name or family to search for
            feature_extractor (bool, optional): Search feature extractors. Defaults to False.
            image_processor (bool, optional): Search image processors. Defaults to False.
            tokenizer (bool, optional): Search tokenizers. Defaults to False.

        Returns:
            tuple: A tuple containing (processor_class, processor_name) where:
                - processor_class: The class to instantiate the processor
                - processor_name: The full name of the matching processor

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If no matching processors found

        Examples:
            >>> # Search for feature extractor
            >>> search("vit", feature_extractor=True)
            (ViTFeatureExtractor, "google/vit-base-patch16-224")

            >>> # Search for image processor
            >>> search("sam", image_processor=True) 
            (SamImageProcessor, "facebook/sam-vit-base")

            >>> # Default search across all types
            >>> search("bert")
            (AutoProcessor, "bert-base-uncased")
        """
        processed_name = self.extract_model_family(query)
        results = []
        
        # Phase 1: Targeted search based on user request
        priority_order = self._get_priority_order(feature_extractor, image_processor, tokenizer)
        
        # Search primary targets
        primary_matches = self._search_mappings(priority_order, processed_name)
        if primary_matches:
            return self._select_best_match(primary_matches, priority_order)
        
        # Phase 2: Fallback to other processor types if no matches found
        fallback_order = [
            "PROCESSOR_MAPPING_NAMES",
            "IMAGE_PROCESSOR_MAPPING_NAMES",
            "FEATURE_EXTRACTOR_MAPPING_NAMES",
            "TOKENIZER_MAPPING_NAMES"
        ]
        fallback_matches = self._search_mappings(fallback_order, processed_name)
        
        return self._select_best_match(fallback_matches, fallback_order) if fallback_matches else (None, None)

    def _get_priority_order(self, fe, ip, tok):

        """Determines search priority order based on processor type flags.

        Internal method used by search() to determine which processor mappings
        to check first based on user-specified flags.

        Args:
            fe (bool): Feature extractor flag
            ip (bool): Image processor flag
            tok (bool): Tokenizer flag

        Returns:
            list: List of mapping names in priority order. Possible values:
                - ["FEATURE_EXTRACTOR_MAPPING_NAMES"]
                - ["IMAGE_PROCESSOR_MAPPING_NAMES"] 
                - ["TOKENIZER_MAPPING_NAMES"]
                - ["PROCESSOR_MAPPING_NAMES", "TOKENIZER_MAPPING_NAMES"]

        Examples:
            >>> _get_priority_order(fe=True, ip=False, tok=False)
            ["FEATURE_EXTRACTOR_MAPPING_NAMES"]
            
            >>> _get_priority_order(fe=False, ip=False, tok=False)
            ["PROCESSOR_MAPPING_NAMES", "TOKENIZER_MAPPING_NAMES"]

        Note:
            This is a private method intended for internal use by the search() method.
        """

        if fe: return ["FEATURE_EXTRACTOR_MAPPING_NAMES"]
        if ip: return ["IMAGE_PROCESSOR_MAPPING_NAMES"]
        if tok: return ["TOKENIZER_MAPPING_NAMES"]
        return ["PROCESSOR_MAPPING_NAMES", "TOKENIZER_MAPPING_NAMES"]


    def _search_mappings(self, mappings, query):
        """Searches through processor mappings for matches to the query.

        Internal method that searches through the provided mapping dictionaries
        for processor names that match the query string.

        Args:
            mappings (list): List of mapping dictionary names to search through.
                Example: ["PROCESSOR_MAPPING_NAMES", "TOKENIZER_MAPPING_NAMES"]
            query (str): The model family name to search for in the mappings.

        Returns:
            list: List of tuples containing (processor_class, processor_name) for
                all matching processors found in the specified mappings.

        Examples:
            >>> mappings = ["PROCESSOR_MAPPING_NAMES"]
            >>> _search_mappings(mappings, "bert")
            [(BertTokenizer, "bert-base-uncased")]

        Note:
            This is a private method intended for internal use by the search() method.
            The search is case-insensitive and matches partial names.
        """
        results = []
        for dict_name in mappings:
            current_dict = self.ordered_dicts_mapping[dict_name]
            
            # Exact match check
            if query in current_dict:
                results.append((dict_name, current_dict[query], 100))
                continue
                
            # Fuzzy match with lower threshold
            for key in current_dict:
                score = fuzz.ratio(query, key.lower())
                if score > 60:  # More lenient threshold
                    results.append((dict_name, current_dict[key], score))
        
        return results

    def _select_best_match(self, matches, priority_order):
        """Selects the best matching processor from a list of matches.

        Internal method that selects the most appropriate processor from multiple
        matches based on the priority order of processor types.

        Args:
            matches (list): List of tuples containing (processor_class, processor_name)
                for all matching processors found.
            priority_order (list): List of mapping names in order of preference.
                Example: ["FEATURE_EXTRACTOR_MAPPING_NAMES", "PROCESSOR_MAPPING_NAMES"]

        Returns:
            tuple: A tuple containing (processor_class, processor_name) for the best
                matching processor, or (None, None) if no matches found.

        Examples:
            >>> matches = [(ViTFeatureExtractor, "vit-base"), (AutoProcessor, "vit-large")]
            >>> priority_order = ["FEATURE_EXTRACTOR_MAPPING_NAMES"]
            >>> _select_best_match(matches, priority_order)
            (ViTFeatureExtractor, "vit-base")

        Note:
            This is a private method intended for internal use by the search() method.
            Selection criteria:
            1. First matching processor type in priority order
            2. First match within that processor type
        """
        scored_matches = sorted(
            matches,
            key=lambda x: (priority_order.index(x[0]), -x[2]),
            reverse=False
        )
        best_match = scored_matches[0]
        return self.model_classes_mapping[best_match[0]], best_match[1]

import os
import subprocess
import json
from pathlib import Path
class appledepth:
    def _init_(self):
        self.model = None
        self.transform = None
        self.image = None
        self.f_px = None
    
    @staticmethod
    def conda_env_exists(env_name):
        """Check if a Conda environment exists."""
        try:
            output = subprocess.check_output(
                ["conda", "env", "list", "--json"],
                text=True,
                stderr=subprocess.PIPE
            )
            envs_data = json.loads(output)
            return any(os.path.basename(env_path) == env_name 
                    for env_path in envs_data.get('envs', []))
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Error checking Conda environments: {e}")
            return False
        
    @staticmethod 
    def is_git_repo(path):
        """Check if a directory is a git repository."""
        return (Path(path) / ".git").exists()
    
    @staticmethod
    def models_exist(repo_dir):
        """Check if the core model file exists."""
        return (Path(repo_dir) / "checkpoints" / "depth_pro.pt").exists()
    
    @staticmethod
    def env_setup():
            # Check if git is installed
        try:
            subprocess.run(
                ["git", "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            print("Error: Git is not installed or not in PATH. Please install Git and try again.")
            return

        # Check if conda is installed
        try:
            subprocess.run(
                ["conda", "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            print("Error: Conda is not installed or not in PATH. Please install Conda (Anaconda/Miniconda) and try again.")
            return
        
        repo_dir = "ml-depth-pro"
        env_name = "depth-pro"

        # Clone or update repository
        if os.path.exists(repo_dir):
            if appledepth.is_git_repo(repo_dir):
                try:
                    print("Updating existing repository...")
                    subprocess.run(
                        ["git", "-C", repo_dir, "pull"],
                        check=True
                    )
                    print("Repository updated successfully.")
                except subprocess.CalledProcessError:
                    print("Warning: Could not update repository. Using existing version.")
            else:
                print("Warning: Existing directory is not a git repository. Using as-is.")
        else:
            try:
                print("Cloning the repository...")
                subprocess.run(
                    ["git", "clone", "https://github.com/apple/ml-depth-pro.git"],
                    check=True
                )
                print("Repository cloned successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error cloning repository: {e}")
                print("Please check your internet connection or the repository URL.")
                return
        
        # Conda environment setup 
        if not appledepth.conda_env_exists(env_name):
            try:
                print(f"Creating Conda environment '{env_name}'...")
                subprocess.run(
                    ["conda", "create", "-n", env_name, "-y", "python=3.9"],
                    check=True
                )
                print("Environment created successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error creating Conda environment: {e}")
                print("Ensure you have sufficient permissions and disk space.")
                return
        else:
            print(f"Conda environment '{env_name}' already exists. Proceeding...")

        # Install package
        try:
            installed = subprocess.run(
                ["conda", "run", "-n", env_name, "pip", "show", "depth-pro"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            ).returncode == 0
            
            if not installed:
                print("Installing package in development mode...")
                subprocess.run(
                    ["conda", "run", "-n", env_name, "pip", "install", "-e", "."],
                    cwd=repo_dir,
                    check=True
                )
                print("Package installed successfully.")
            else:
                print("Package already installed. Skipping installation.")
        except subprocess.CalledProcessError as e:
            print(f"Error checking/installing package: {e}")
            return

        # Model download handling
        model_path = Path(repo_dir) / "checkpoints" / "depth_pro.pt"
        script_path = Path(repo_dir) / "get_pretrained_models.sh"
        
        if not model_path.exists():
            try:
                print("Downloading pretrained models...")
                
                # Ensure checkpoints directory exists
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Clean up any previous partial downloads
                for f in model_path.parent.glob("depth_pro.pt*"):
                    print(f"Removing previous download: {f}")
                    f.unlink()

                # Make script executable and run (CORRECTED LINE)
                subprocess.run(["chmod", "+x", str(script_path)], check=True)
                subprocess.run(
                    ["./get_pretrained_models.sh"],  # This was the critical fix
                    cwd=repo_dir,
                    check=True
                )
                
                # Final verification
                if not model_path.exists():
                    raise RuntimeError("Model download failed - check repo's get_pretrained_models.sh script")
                    
                print("Models downloaded successfully.")
            except (subprocess.CalledProcessError, RuntimeError) as e:
                print(f"Error downloading models: {e}")
                return
        else:
            print("Pretrained models already exist. Skipping download.")

        print("Environment setup complete.")
            

    def load_model(self):
        """Loads and initializes the depth estimation model.

        This method creates the model and its associated transforms, then sets
        the model to evaluation mode. It uses the depth_pro module from the src
        package to create the model instance.

        Returns:
            None

        Raises:
            ImportError: If the src.depth_pro module cannot be imported.
            RuntimeError: If model creation or initialization fails.
            Exception: If model creation fails.

        Example:
            model_instance = ModelClass()
            model_instance.load_model()
        """
        try:
            from src import depth_pro
            self.model, self.transform = depth_pro.create_model_and_transforms()
            self.model.eval()
        except ImportError as e:
            raise ImportError(f"Failed to import depth_pro module: {str(e)}")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error while loading model: {str(e)}")

    def processor(self, image_path,text = None):
        """Processes input image for depth estimation.
    
        This method loads and preprocesses the RGB image using depth_pro module's
        transform pipeline. The processed image and focal length are stored as
        instance variables.

        Args:
            image_path (str): Path to the input RGB image file.
            text (str, optional): Text prompt if model supports text guidance. Defaults to None.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ImportError: If depth_pro module cannot be imported.
            RuntimeError: If image processing fails.

        Side Effects:
            Sets the following instance variables:
                - self.image: Transformed image tensor ready for inference
                - self.f_px: Focal length in pixels
        
        Example:
            model = AppleDepth()
            model.processor("/path/to/image.jpg")
        """
        from src import depth_pro
        self.image, _, self.f_px = depth_pro.load_rgb(image_path)
        self.image = self.transform(self.image)

    def infer(self):
        """Performs depth estimation inference on the processed image.

        This method uses the loaded model to perform depth estimation on the 
        preprocessed image stored in self.image.

        Prerequisites:
            - Model must be loaded via load_model()
            - Image must be processed via processor()
            - self.image and self.f_px must be set

        Returns:
            torch.Tensor: Predicted depth map with shape (H, W)

        Raises:
            RuntimeError: If model is not loaded or image is not processed
            ValueError: If focal length is invalid
            Exception: If inference fails

        Example:
            model = AppleDepth()
            model.load_model()
            model.processor("image.jpg")
            depth_map = model.infer()
        """
        prediction = self.model.infer(self.image, f_px=self.f_px)
        return prediction

# class Marigold:
#     def _init_(self):
#         #**fill in this**
#         self.pipe = None
#         self.image = None
#         self.depth = None

#     @staticmethod
#     def env_setup():
#         #**fill in this**
#         pass

#     def load_model(self):
#         #**fill in this**
#         #import torch
#         import diffusers
#         self.pipe = diffusers.MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-lcm-v0-1")

#     def processor(self, image_path,text = None):
#         #**fill in this**
#         import diffusers
#         self.image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
#         self.depth = self.pipe(self.image)


#     def infer(self):
#         #**fill in this**
#         self.depth = self.pipe.image_processor.visualize_normals(self.depth.prediction)


#if some function is not needed then it shold not be removed but should be filled with pass


reference_table =  {
                                        "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES": AutoModelForCausalLM,
                                        "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES": AutoModelForImageClassification,
                                        "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES": AutoModelForSemanticSegmentation,
                                        "MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES": AutoModelForUniversalSegmentation,
                                        "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES": AutoModelForVideoClassification,
                                        "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES": AutoModelForVision2Seq,
                                        "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES": AutoModelForImageTextToText,
                                        "MODEL_FOR_MASKED_LM_MAPPING_NAMES": AutoModelForMaskedLM,
                                        "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES": AutoModelForObjectDetection,
                                        "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES": AutoModelForZeroShotObjectDetection,
                                        "MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES": AutoModelForDepthEstimation,
                                        "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES": AutoModelForSeq2SeqLM,
                                        "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES": AutoModelForSpeechSeq2Seq,
                                        "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES": AutoModelForSequenceClassification,
                                        "MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES": AutoModelForQuestionAnswering,
                                        "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES": AutoModelForDocumentQuestionAnswering,
                                        "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES": AutoModelForTableQuestionAnswering,
                                        "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES": AutoModelForTokenClassification,
                                        "MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES": AutoModelForMultipleChoice,
                                        "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES": AutoModelForNextSentencePrediction,
                                        "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES": AutoModelForAudioClassification,
                                        "MODEL_FOR_CTC_MAPPING_NAMES": AutoModelForCTC,
                                        "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES": AutoModelForAudioFrameClassification,
                                        "MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES": AutoModelForAudioXVector,
                                        "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES": AutoModelForTextToSpectrogram,
                                        "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES": AutoModelForTextToWaveform,
                                        "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES": AutoModelForZeroShotImageClassification,
                                        "MODEL_FOR_MASK_GENERATION_MAPPING_NAMES": AutoModelForMaskGeneration,
                                        "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES": AutoModelForVisualQuestionAnswering,
                                        "AppledepthPro" : appledepth,
                                        # "prs-eth/marigold-normals-lcm-v0-1" : Marigold
                                      }


