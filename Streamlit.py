import streamlit as st
from PIL import Image
import base64
from UniVLM.Model import Yggdrasil

# 🔥 Cyberpunk CSS Injection
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Ubuntu+Mono&display=swap');
    
    .stApp {{
        background: linear-gradient(135deg, #0a0a2e 0%, #1a1a4a 100%);
        color: #00ff9d !important;
        font-family: 'Ubuntu Mono', monospace;
    }}
    
    .cyber-header {{
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 10px #00ff9d;
        border-bottom: 3px solid #ff003c;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }}
    
    .stButton>button {{
        background: linear-gradient(45deg, #ff003c, #ff00ff);
        color: #000 !important;
        border: 1px solid #00ff9d !important;
        border-radius: 0 !important;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
        transition: 0.3s !important;
    }}
    
    .stButton>button:hover {{
        box-shadow: 0 0 15px #ff003c;
        transform: skewX(-10deg);
    }}
    
    .cyber-card {{
        background: rgba(0, 0, 30, 0.9);
        border: 1px solid #00ff9d;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }}
    
    .cyber-card::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, #00ff9d33, transparent);
        transform: rotate(45deg);
        animation: scan 5s linear infinite;
    }}
    
    @keyframes scan {{
        0% {{ transform: translateY(-100%) rotate(45deg); }}
        100% {{ transform: translateY(100%) rotate(45deg); }}
    }}
</style>
""", unsafe_allow_html=True)

def cyber_header():
    st.markdown("""
    <div class="cyber-header">
        <h1>⏣ YGGDRASIL NEURAL INTERFACE</h1>
        <h3>// MULTI-MODAL INFERENCE ENGINE v2.3.5</h3>
    </div>
    """, unsafe_allow_html=True)

def model_configuration():
    with st.container():
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.markdown("### 💾 CORE CONFIGURATION")
        
        cols = st.columns([3,2])
        with cols[0]:
            model_name = st.text_input("NEURAL MATRIX ID", "gpt2", 
                                     help="Input protocol identifier")
        with cols[1]:
            config_name = st.text_input("CONFIG PRESET", "default",
                                      help="Activation schema")
            
        feature_extractor = st.text_input("SIGNATURE ANALYZER", "auto-detect")
        image_processor = st.text_input("OPTICS DECODER", "auto-detect")
        
        if st.button("⚡ ACTIVATE CORE"):
            try:
                st.session_state.model = Yggdrasil(
                    model_name=model_name,
                    Feature_extractor=feature_extractor,
                    Image_processor=image_processor,
                    Config_Name=config_name
                )
                load_status = st.session_state.model.load()
                
                if load_status == "Loaded":
                    st.success(f"✅ CORE ONLINE :: {st.session_state.model.model_type.upper()} PROTOCOL")
                    if st.session_state.model.model_type == "HF":
                        st.session_state.model.Processor()
                        st.experimental_rerun()
                else:
                    st.error("❌ CORE FAILURE :: CHECK NEURAL LINK")
            except Exception as e:
                st.error(f"💥 SYSTEM ERROR :: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

def inference_interface():
    if 'model' not in st.session_state or not st.session_state.model:
        st.warning("⚠️ INITIALIZE NEURAL CORE FIRST")
        return
    
    with st.container():
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.markdown("### 🖥️ RUNTIME CONSOLE")
        
        input_type = st.radio("INPUT MODALITY", 
                            ["📟 TEXT STREAM", "📡 VISUAL FEED", "📡 MULTI-SPECTRAL"], 
                            horizontal=True)
        
        payload = {}
        cols = st.columns([3,2])
        with cols[0]:
            if "TEXT" in input_type or "MULTI" in input_type:
                payload["prompt"] = st.text_area("NEURAL INPUT", 
                                               "INITIALIZING DATA STREAM...",
                                               height=150)
        with cols[1]:
            if "VISUAL" in input_type or "MULTI" in input_type:
                uploaded_file = st.file_uploader("UPLOAD VISUAL DATA", 
                                               type=["jpg", "png"],
                                               help="Supported formats: PNG/JPG")
                if uploaded_file:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption='VISUAL FEED ACQUIRED', width=200)
                    payload["images"] = image
        
        if st.button("🚀 EXECUTE INFERENCE", key="run"):
            with st.spinner("ANALYZING NEURAL PATTERNS..."):
                try:
                    results = st.session_state.model.inference(payload)
                    display_results(results)
                except Exception as e:
                    st.error(f"💥 RUNTIME FAILURE :: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

def display_results(results):
    st.markdown("---")
    st.markdown("### 🔄 OUTPUT STREAM")
    
    if isinstance(results, list):
        for i, res in enumerate(results):
            st.markdown(f"**📊 DATA CHUNK {i+1}**")
            if isinstance(res, Image.Image):
                st.image(res, use_column_width=True)
            else:
                st.markdown(f"```cyber\n{res}\n```")
    else:
        if isinstance(results, Image.Image):
            st.image(results, use_column_width=True)
        else:
            st.markdown(f"```cyber\n{results}\n```")

def main():
    cyber_header()
    model_configuration()
    inference_interface()

if __name__ == "__main__":
    main()