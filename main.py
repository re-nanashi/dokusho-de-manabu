import whisper
import torch
import pysbd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = whisper.load_model("medium").to(device)
result = model.transcribe("./3-sample.wav", language="japanese")

seg = pysbd.Segmenter(language="ja", clean=False)

extracted_jp_text = result["text"]
bun_list = seg.segment(extracted_jp_text)

for bun in bun_list:
    print(bun)
