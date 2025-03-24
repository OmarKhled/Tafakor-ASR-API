import argparse
from nemo.collections.asr.models import ASRModel
from pyctcdecode import build_ctcdecoder
import numpy as np
import correction

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
    return " - ".join(corpus[surah_index]["verses"][start_index - 1:end_index])

def main(args):
    labels = open("labels.txt", "r").read().split("\n")
    quran_uthmani = open("./quran-uthmani.txt", "r").read().split("\n")

    corpus = decode_corpus(quran_uthmani)
    ground_truth = verses_segment(corpus, args.surah_number, args.start_verse, args.end_verse)

    audio_path = args.audio_file

    asr_model = ASRModel.restore_from("./nemo_model/conformer_ctc_char_small_dataset_v2.nemo")

    log_probs = asr_model.transcribe(audio_path, return_hypotheses=True)
    logits = [hyp.alignments for hyp in log_probs][0]

    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path="./ngrams/ngram_model_8.bin.tmp.arpa",
        alpha=0.5,
        beta=1.0,
    )

    transcription = decoder.decode(np.array(logits))

    print("Transcription:")
    print(transcription)

    print("----")

    corrected_transcription = correction.refine_transcription(transcription, ground_truth)

    print("Corrected Transcription:")
    print(corrected_transcription)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Quran Transcription and Correction")
    parser.add_argument('--audio_file', type=str, required=True, help='Path to the audio file')
    parser.add_argument('--surah_number', type=int, required=True, help='Surah number to transcribe')
    parser.add_argument('--start_verse', type=int, required=True, help='Starting verse number')
    parser.add_argument('--end_verse', type=int, required=True, help='Ending verse number')

    args = parser.parse_args()

    main(args)