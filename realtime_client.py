import asyncio
import base64
import json
import queue
from typing import Any

import numpy as np
import sounddevice as sd
import websockets
from scipy.signal import resample_poly
import time

WS_URL = "ws://127.0.0.1:8765/v1/realtime"

INPUT_DEVICE = 44          # NVIDIA Broadcast, WASAPI
OUTPUT_DEVICE = 27         # RX-V4A, WASAPI

MIC_SAMPLE_RATE = 48000
SERVER_INPUT_RATE = 16000
TTS_SAMPLE_RATE = 24000
OUTPUT_DEVICE_RATE = 48000

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 2
BLOCK_SIZE = 1440          # 30 ms @ 48 kHz

mic_queue: queue.Queue[bytes] = queue.Queue()
last_meter_time = 0.0

def microphone_callback(
    indata: np.ndarray,
    frames: int,
    time_info: Any,
    status: sd.CallbackFlags,
) -> None:
    global last_meter_time

    if status:
        print(f"[麥克風] {status}")

    mono_48k = np.asarray(indata[:, 0], dtype=np.float32)

    rms = float(np.sqrt(np.mean(mono_48k ** 2)))

    now = time.monotonic()
    if now - last_meter_time >= 1.0:
        #print(f"[麥克風音量] RMS={rms:.5f}")
        last_meter_time = now

    mono_16k = resample_poly(mono_48k, up=1, down=3)

    pcm16 = np.clip(mono_16k, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    mic_queue.put(pcm16.tobytes())


async def send_microphone(ws: Any) -> None:
    packets = 0
    last_report = time.monotonic()

    while True:
        try:
            audio_bytes = mic_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio_bytes).decode("ascii"),
                }
            )
        )

        packets += 1

        now = time.monotonic()
        if now - last_report >= 1.0:
            #print(f"[上傳音訊] {packets} packets/s")
            packets = 0
            last_report = now


def convert_tts_audio_for_output(pcm_bytes: bytes) -> bytes:
    mono_24k = np.frombuffer(
        pcm_bytes,
        dtype=np.int16,
    ).astype(np.float32) / 32768.0

    # 24 kHz → 48 kHz
    mono_48k = resample_poly(mono_24k, up=2, down=1)

    # mono → stereo
    stereo_48k = np.column_stack((mono_48k, mono_48k))

    stereo_pcm16 = np.clip(stereo_48k, -1.0, 1.0)
    stereo_pcm16 = (stereo_pcm16 * 32767.0).astype(np.int16)

    return stereo_pcm16.tobytes()


async def receive_events(
    ws: Any,
    output_stream: sd.RawOutputStream,
) -> None:
    async for raw in ws:
        event = json.loads(raw)
        event_type = event.get("type", "")

        if event_type == "session.created":
            print("已連線到 Realtime 服務")

        elif event_type == "input_audio_buffer.speech_started":
            print("\n[開始說話]")

        elif event_type == "input_audio_buffer.speech_stopped":
            print("[停止說話]")

        elif event_type == (
            "conversation.item.input_audio_transcription.completed"
        ):
            print(f"\n你：{event.get('transcript', '')}")

        elif event_type == "response.output_audio_transcript.done":
            print(f"助理：{event.get('transcript', '')}")

        elif event_type == "response.output_audio.delta":
            delta = event.get("delta")
            if delta:
                pcm_24k = base64.b64decode(delta)
                output_bytes = convert_tts_audio_for_output(pcm_24k)
                output_stream.write(output_bytes)

        elif event_type == "response.done":
            print("[本輪完成]")

        elif event_type == "error":
            print(
                "[伺服器錯誤]",
                json.dumps(event, ensure_ascii=False, indent=2),
            )


async def main() -> None:
    print(f"連線至 {WS_URL}")

    async with websockets.connect(
        WS_URL,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    ) as ws:
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": (
                    "你是一位語音助理，請使用臺灣繁體中文回答，"
                    "回答自然簡短，不要使用 Markdown。"
                ),
                "turn_detection": {
                    "type": "server_vad",
                    "interrupt_response": True,
                },
            },
        }

        await ws.send(
            json.dumps(session_update, ensure_ascii=False)
        )

        with sd.InputStream(
            device=INPUT_DEVICE,
            samplerate=MIC_SAMPLE_RATE,
            channels=INPUT_CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=microphone_callback,
        ), sd.RawOutputStream(
            device=OUTPUT_DEVICE,
            samplerate=OUTPUT_DEVICE_RATE,
            channels=OUTPUT_CHANNELS,
            dtype="int16",
        ) as output_stream:
            print("可以開始說話，按 Ctrl+C 結束。")

            await asyncio.gather(
                send_microphone(ws),
                receive_events(ws, output_stream),
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已結束")
    except websockets.exceptions.ConnectionClosedError as exc:
        print(f"\n連線被伺服器關閉：{exc}")