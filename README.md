# Orpheus-FastAPI

Text-to-Speech server with OpenAI-compatible API, 8 voices, and emotion tags; optimized for RTX GPUs

## Model Collection

- **Q2_K**: Ultra-fast inference with 2-bit quantization
- **Q4_K_M**: Balanced quality/speed with 4-bit quantization (mixed)
- **Q8_0**: Original high-quality 8-bit model

[Browse the Orpheus-FASTAPI Model Collection on HuggingFace](https://huggingface.co/collections/lex-au/orpheus-fastapi-67e125ae03fc96dae0517707)

## Setup

### Prerequisites

- Python 3.8-3.11 (**Python 3.12 is not supported due to removal of pkgutil.ImpImporter!!**)
- CUDA-compatible GPU (recommended: RTX series for best performance)
- Separate LLM inference server running the Orpheus model (e.g., LM Studio or llama.cpp server)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Lex-au/Orpheus-FastAPI.git
cd Orpheus-FastAPI
```

2. Create a Python virtual environment:
```bash
# Using venv (Python's built-in virtual environment)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install PyTorch:
```bash
# normal
pip3 install torch torchaudio
# from nvidia
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

4. Install other dependencies:
```bash
pip3 install -r requirements.txt
```

5. Install dev deps if necessary:
```bash
pip3 install -r requirements.dev.txt
```

6. Start the server:
```bash
uvicorn app:app --host 0.0.0.0 --port 5005 --reload
```

## API Usage

API documentation can be found locally at `/docs`.

### OpenAI-Compatible Endpoint

The server provides an OpenAI-compatible API endpoint at `/v1/audio/speech`:

```bash
curl http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus",
    "input": "Hello world! This is a test of the Orpheus TTS system.",
    "voice": "tara",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

### Parameters

- `input` (required): The text to convert to speech
- `model` (optional): The model to use (default: "orpheus")
- `voice` (optional): Which voice to use (default: "tara")
- `response_format` (optional): Output format (currently only "wav" is supported)
- `speed` (optional): Speed factor (0.5 to 1.5, default: 1.0)

### Available Voices

- `tara`: Female, conversational, clear
- `leah`: Female, warm, gentle
- `jess`: Female, energetic, youthful
- `leo`: Male, authoritative, deep
- `dan`: Male, friendly, casual
- `mia`: Female, professional, articulate
- `zac`: Male, enthusiastic, dynamic
- `zoe`: Female, calm, soothing

### Emotion Tags

You can insert emotion tags into your text to add expressiveness:

- `<laugh>`: Add laughter
- `<sigh>`: Add a sigh
- `<chuckle>`: Add a chuckle
- `<cough>`: Add a cough sound
- `<sniffle>`: Add a sniffle sound
- `<groan>`: Add a groan
- `<yawn>`: Add a yawning sound
- `<gasp>`: Add a gasping sound

Example: `"Well, that's interesting <laugh> I hadn't thought of that before."`

## Technical Details

This server works as a frontend that connects to an external LLM inference server. It sends text prompts to the inference server, which generates tokens that are then converted to audio using the SNAC model. The system has been optimised for RTX 4090 GPUs with:

- Vectorised tensor operations
- Parallel processing with CUDA streams
- Efficient memory management
- Token and audio caching
- Optimised batch sizes

### Hardware Detection and Optimization

The system features intelligent hardware detection that automatically optimizes performance based on your hardware capabilities:

- **High-End GPU Mode** (dynamically detected based on capabilities):
  - Triggered by either: 16GB+ VRAM, compute capability 8.0+, or 12GB+ VRAM with 7.0+ compute capability
  - Advanced parallel processing with 4 workers
  - Optimized batch sizes (32 tokens)
  - High-throughput parallel file I/O
  - Full hardware details displayed (name, VRAM, compute capability)
  - GPU-specific optimizations automatically applied

- **Standard GPU Mode** (other CUDA-capable GPUs):
  - Efficient parallel processing
  - GPU-optimized parameters
  - CUDA acceleration where beneficial
  - Detailed GPU specifications

- **CPU Mode** (when no GPU is available):
  - Conservative processing with 2 workers
  - Optimized memory usage
  - Smaller batch sizes (16 tokens)
  - Sequential file I/O
  - Detailed CPU cores, threads, and RAM information

No manual configuration is needed - the system automatically detects hardware capabilities and adapts for optimal performance across different generations of GPUs and CPUs.

### Token Processing Optimization

The token processing system has been optimized with mathematically aligned parameters:
- Uses a context window of 49 tokens (7²)
- Processes in batches of 7 tokens (Orpheus model standard)
- This square relationship ensures complete token processing with no missed tokens
- Results in cleaner audio generation with proper token alignment
- Repetition penalty fixed at 1.1 for optimal quality generation (cannot be changed)

### Long Text Processing

The system features efficient batch processing for texts of any length:
- Automatically detects longer inputs (>1000 characters) 
- Splits text at logical points to create manageable chunks
- Processes each chunk independently for reliability
- Combines audio segments with smooth 50ms crossfades
- Intelligently stitches segments in-memory for consistent output
- Handles texts of unlimited length with no truncation
- Provides detailed progress reporting for each batch

**Note about long-form audio**: While the system now supports texts of unlimited length, there may be slight audio discontinuities between segments due to architectural constraints of the underlying model. The Orpheus model was designed for short to medium text segments, and our batching system works around this limitation by intelligently splitting and stitching content with minimal audible impact.

### External Inference Server

This application requires a separate LLM inference server running the Orpheus model. You can use:

- [GPUStack](https://github.com/gpustack/gpustack) - GPU optimised LLM inference server (My pick) - supports LAN/WAN tensor split parallelisation
- [LM Studio](https://lmstudio.ai/) - Load the GGUF model and start the local server
- [llama.cpp server](https://github.com/ggerganov/llama.cpp) - Run with the appropriate model parameters
- Any compatible OpenAI API-compatible server

**Quantized Model Options:**
- **lex-au/Orpheus-3b-FT-Q2_K.gguf**: Fastest inference (~50% faster tokens/sec than Q8_0)
- **lex-au/Orpheus-3b-FT-Q4_K_M.gguf**: Balanced quality/speed 
- **lex-au/Orpheus-3b-FT-Q8_0.gguf**: Original high-quality model

Choose based on your hardware and needs. Lower bit models (Q2_K, Q4_K_M) provide ~2x realtime performance on high-end GPUs.

[Browse all models in the collection](https://huggingface.co/collections/lex-au/orpheus-fastapi-67e125ae03fc96dae0517707)

The inference server should be configured to expose an API endpoint that this FastAPI application will connect to.

### Environment Variables

You can configure the system using environment variables or a `.env` file:

- `ORPHEUS_API_URL`: URL of the LLM inference API (tts_engine/inference.py)
- `ORPHEUS_API_TIMEOUT`: Timeout in seconds for API requests (default: 120)
- `ORPHEUS_MAX_TOKENS`: Maximum tokens to generate (default: 8192)
- `ORPHEUS_TEMPERATURE`: Temperature for generation (default: 0.6)
- `ORPHEUS_TOP_P`: Top-p sampling parameter (default: 0.9)
- `ORPHEUS_SAMPLE_RATE`: Audio sample rate in Hz (default: 24000)
- `ORPHEUS_PORT`: Web server port (default: 5005)
- `ORPHEUS_HOST`: Web server host (default: 0.0.0.0)

The system now supports loading environment variables from a `.env` file in the project root, making it easier to configure without modifying system-wide environment settings. See `.env.example` for a template.

> [!NOTE]
> Repetition penalty is hardcoded to 1.1 as this is the only value that produces stable, high-quality output.

## Development

### Adding New Voices

To add new voices, update the `AVAILABLE_VOICES` list in `tts_engine/inference.py` and add corresponding descriptions in the HTML template.

## Using with llama.cpp

When running the Orpheus model with llama.cpp, use these parameters to ensure optimal performance:

```bash
./llama-server -m models/Modelname.gguf \
  --ctx-size {{your ORPHEUS_MAX_TOKENS from .env}} \
  --n-predict {{your ORPHEUS_MAX_TOKENS from .env}} \
  --rope-scaling linear \
  <other options>
```

Where:
- `--ctx-size`: Sets the context window size, should match your ORPHEUS_MAX_TOKENS setting
- `--n-predict`: Maximum tokens to generate, should match your ORPHEUS_MAX_TOKENS setting
- `--rope-scaling linear`: Required for optimal positional encoding with the Orpheus model

For extended audio generation (books, long narrations), you may want to increase your token limits:
1. Set ORPHEUS_MAX_TOKENS to 32768 or higher in your .env file (or via the Web UI)
2. Increase ORPHEUS_API_TIMEOUT to 1800 for longer processing times
3. Use the same values in your llama.cpp parameters (if you're using llama.cpp)
