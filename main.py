import whisper
import pysbd
import torch
import time
import wave
import contextlib
from faster_whisper import WhisperModel

SECONDS_PER_MINUTE = 60
DEFAULT_BEAM_SIZE = 5
COMPUTE_TYPE = "float16"
MODEL = "medium"

audio_fname = "./Sep 02-06.wav"
with contextlib.closing(wave.open(audio_fname, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    duration_minutes = duration // SECONDS_PER_MINUTE
    print(f"Audio duration: {duration_minutes} minutes")

start_time = time.time()

# Get the device then print for logging purposes
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device that will be used for extraction: {device}")
print("Starting extraction...")
# faster-whisper:
model = WhisperModel(MODEL, device="cuda", compute_type=COMPUTE_TYPE)
segments, _ = model.transcribe(audio_fname, beam_size=DEFAULT_BEAM_SIZE)


file_name = "output.txt"
with open(file_name, 'w') as out:
    for segment in segments:
        line = "[%.2fs -> %.2fs] %s" % (segment.start,
                                        segment.end, segment.text)
        print(line)
        out.write("\n" + line)

overall_duration = (time.time() - start_time) / 60
print("--- %s minutes ---" % (overall_duration))
