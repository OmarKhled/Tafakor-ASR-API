import os
import tempfile
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from nemo.collections.asr.models import ASRModel
from pyctcdecode import build_ctcdecoder
import correction

# FastAPI app initialization
app = FastAPI()

class QuranRequest(BaseModel):
    surah_number: int
    start_verse: int
    end_verse: int

def decode_line(line: str):
    [surah_number, verse_number, verse_text] = line.split("|")
    return int(surah_number), int(verse_number), verse_text

def decode_corpus(quran_corpus: str):
    corpus = {}
    for index, line in enumerate(quran_corpus):
        [surah_number, verse_number, verse_text] = decode_line(line)

        if index + 1 == len(quran_corpus) or decode_line(quran_corpus[index+1])[0] == surah_number + 1:
            start_index = index - verse_number + 1
            end_index = index

            corpus[surah_number] = {
                "length": verse_number,
                "verses": [decode_line(line)[2] for line in quran_corpus[start_index: end_index + 1]]
            }
    return corpus

def verses_segment(corpus: dict, surah_index: int, start_index: int, end_index: int):
    return " ".join(corpus[surah_index]["verses"][start_index - 1:end_index])

@app.post("/transcribe/")
async def transcribe_quran(surah_number: int = Form(), start_verse: int = Form(), end_verse: int = Form(), audio_file: UploadFile = File(...)):
    # Save the uploaded audio file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        temp_audio_file.write(await audio_file.read())
        temp_audio_path = temp_audio_file.name
    
    try:
        # Load labels and Quran text files
        labels = open("labels.txt", "r").read().split("\n")
        quran_uthmani = open("./quran-uthmani.txt", "r").read().split("\n")
        
        # Decode the Quran corpus and prepare the ground truth for the requested surah and verses
        corpus = decode_corpus(quran_uthmani)
        ground_truth = verses_segment(corpus, surah_number, start_verse, end_verse)

        # Load ASR model
        asr_model = ASRModel.restore_from("./nemo_model/conformer_ctc_char_small_dataset_v2.nemo")

        print(temp_audio_path)
        # Transcribe audio
        log_probs = asr_model.transcribe(temp_audio_path, return_hypotheses=True)
        logits = [hyp.alignments for hyp in log_probs][0]

        # Build decoder
        decoder = build_ctcdecoder(
            labels,
            kenlm_model_path="./ngrams/ngram_model_8.bin.tmp.arpa",
            alpha=0.5,
            beta=1.0,
        )

        # Decode the transcription
        transcription = decoder.decode(np.array(logits))

        # Refine transcription based on ground truth
        corrected_transcription = correction.refine_transcription(transcription, ground_truth)

        # Return both original and corrected transcription
        return {
            "transcription": transcription,
            "corrected_transcription": corrected_transcription
        }

    finally:
        # Clean up: remove the temporary audio file after processing
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

