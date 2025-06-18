"""
Live transcription demo for OpenAI Realtime API.
Captures from VB-Cable "CABLE Output" at 16 kHz mono PCM.
"""

import asyncio, json, os, base64, sounddevice as sd, numpy as np, websockets
print("websockets imported from:", websockets.__file__)

API_KEY   = os.getenv("OPENAI_API_KEY")
MODEL     = "gpt-4o-mini-realtime-preview"      # or other realtime-enabled model
URI       = "wss://api.openai.com/v1/realtime"  # change if Azure
SAMPLE_RT = 16_000
CHUNK_MS  = 160            # ≈0.16 s latency
CHUNK_SAMPLES = SAMPLE_RT * CHUNK_MS // 1000

# --- helper: locate VB-Cable device index (Windows) --------------------------
def vb_cable_index():
    for i, dev in enumerate(sd.query_devices()):
        if "CABLE Output" in dev['name'] and dev['max_input_channels'] > 0:
            return i
    raise RuntimeError("VB-Audio CABLE Output device not found")

# --- Real-time loop ----------------------------------------------------------
async def realtime_transcribe():
    # Create connection with required headers
    async with websockets.connect(
        URI,
        additional_headers={
            "Authorization": f"Bearer {API_KEY}",
            "OpenAI-Beta": "assistants=v1"  # Updated beta header value
        },
        max_size=1_000_000   # allow large frames
    ) as ws:

        # 1️⃣  tell the server what we're sending
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["audio"],
                "model": MODEL,
                "input_audio_format": "pcm_f16le",   # 16-bit PCM little-endian
                "input_audio_transcription": { "model": "whisper-1" },
                # disable server VAD if you prefer to commit manually
                "turn_detection": { "type": "server" }
            }
        }))

        # 2️⃣  fire up microphone capture
        print("🎙  Listening...  Ctrl-C to quit.")
        stream = sd.InputStream(
            device=vb_cable_index(),
            samplerate=SAMPLE_RT,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SAMPLES
        )
        stream.start()

        async def reader():
            async for message in ws:
                event = json.loads(message)
                if event.get("type") == "response.audio_transcript.delta":
                    print(event["delta"], end="", flush=True)

        async def writer():
            while True:
                audio = stream.read(CHUNK_SAMPLES)[0].flatten()
                pcm_i16 = (audio * 32767).astype(np.int16).tobytes()
                b64 = base64.b64encode(pcm_i16).decode("ascii")
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": b64
                }))

        await asyncio.gather(reader(), writer())

try:
    asyncio.run(realtime_transcribe())
except KeyboardInterrupt:
    print("\n🛑 stopped")
