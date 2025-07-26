import math
import sys
import torch
from transformers import pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import argparse
from tqdm import tqdm
from loguru import logger
import shutil
from pathlib import Path
import warnings


def setup_pipeline(model_id="openai/whisper-large-v3", language="russian"):
    """Setup the ASR pipeline with optimal settings."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info(f"Using device: {device}")
    logger.info(f"Using dtype: {torch_dtype}")

    # Suppress specific transformers warnings that are not actionable
    warnings.filterwarnings("ignore", message=".*The input name `inputs` is deprecated.*")
    warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,  # Optimize chunk length for processing
        stride_length_s=5,  # Add stride for better accuracy at chunk boundaries
    )

    generate_kwargs = {
        "task": "transcribe",
        "language": language,
        # "return_timestamps": "word",
        # Add generation parameters to improve quality and reduce warnings
        # "condition_on_prev_tokens": False,  # Helps with attention mask issues
        # "compression_ratio_threshold": 2.4,  # Helps detect poor quality audio
        # "logprob_threshold": -1.0,  # Helps with confidence filtering
    }

    return asr_pipeline, generate_kwargs


def detect_optimal_silence_threshold(audio, initial_thresh=-45):
    """
    Dynamically detect optimal silence threshold based on audio characteristics.

    Args:
        audio: AudioSegment object
        initial_thresh: Starting threshold

    Returns:
        int: Optimal silence threshold in dBFS
    """
    # Calculate audio statistics
    rms_values = []
    chunk_size_ms = 1000  # 1 second chunks for analysis

    for i in range(0, len(audio), chunk_size_ms):
        chunk = audio[i:i + chunk_size_ms]
        if len(chunk) > 0:
            rms_values.append(chunk.rms)

    if not rms_values:
        return initial_thresh

    # Convert RMS to dBFS approximation
    max_rms = max(rms_values)
    avg_rms = sum(rms_values) / len(rms_values)

    # Calculate percentiles for better threshold detection
    sorted_rms = sorted(rms_values)
    percentile_10 = sorted_rms[int(len(sorted_rms) * 0.1)]
    percentile_25 = sorted_rms[int(len(sorted_rms) * 0.25)]

    # Convert to dBFS (approximate)
    def rms_to_dbfs(rms_val):
        if rms_val <= 0:
            return -80
        return 20 * math.log10(rms_val / max_rms) if max_rms > 0 else -80

    # Calculate adaptive threshold
    silence_dbfs_10 = rms_to_dbfs(percentile_10)
    silence_dbfs_25 = rms_to_dbfs(percentile_25)

    # Choose threshold based on audio characteristics
    if silence_dbfs_25 < -60:  # Very quiet audio
        adaptive_thresh = max(initial_thresh, silence_dbfs_25 + 10)
    elif silence_dbfs_10 > -30:  # Very loud audio
        adaptive_thresh = min(initial_thresh, silence_dbfs_10 - 15)
    else:  # Normal audio
        adaptive_thresh = (silence_dbfs_25 + initial_thresh) / 2

    # Clamp to reasonable range
    adaptive_thresh = max(-80, min(-20, adaptive_thresh))

    logger.debug(
        f"Audio analysis: 10th percentile={silence_dbfs_10:.1f}dBFS, 25th percentile={silence_dbfs_25:.1f}dBFS")
    logger.debug(f"Adaptive threshold: {initial_thresh} → {adaptive_thresh}")

    return int(adaptive_thresh)


def get_audio_chunks(audio_file_path, min_silence_len=700, silence_thresh=-45, max_chunk_len_s=29, min_chunk_len_s=2):
    """
    Splits an audio file into chunks based on silence, ensuring no chunk exceeds max_chunk_len_s.

    Args:
        audio_file_path (str): Path to the audio file.
        min_silence_len (int): Minimum silence length in ms to split on.
        silence_thresh (int): Silence threshold in dBFS.
        max_chunk_len_s (int): Maximum length of a chunk in seconds.

    Returns:
        list: A list of pydub AudioSegment chunks.
    """
    logger.info("Loading audio and splitting by silence...")
    audio = AudioSegment.from_file(audio_file_path)

    silence_thresh=detect_optimal_silence_threshold(audio,initial_thresh=silence_thresh)

    # Split audio by silence
    audio_len=len(audio)
    audio_len_s=audio_len/ 1000
    split_bar = tqdm(desc="Processing audio",total=math.ceil(audio_len_s),unit="s")
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=400,  # Keep a bit of silence at the ends of chunks
        callback=lambda t: split_bar.update(math.ceil(t/1000)-split_bar.n)
    )
    split_bar.close()

    # Filter out chunks that are too small
    min_chunk_len_ms = min_chunk_len_s * 1000
    filtered_chunks = []
    dropped_count = 0

    for chunk in chunks:
        if len(chunk) < min_chunk_len_ms:
            dropped_count += 1
            logger.debug(f"Dropping small chunk: {len(chunk) / 1000:.2f}s")
        else:
            filtered_chunks.append(chunk)

    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} chunks smaller than {min_chunk_len_s}s")

    # If a chunk is too long, split it further
    final_chunks = []
    max_chunk_len_ms = max_chunk_len_s * 1000

    for chunk in filtered_chunks:
        if len(chunk) > max_chunk_len_ms:
            logger.warning(f"Chunk exceeds max length ({len(chunk) / 1000:.1f}s), splitting...")
            # Split the long chunk into smaller pieces
            subchunks = [chunk[i:i + max_chunk_len_ms] for i in range(0, len(chunk), max_chunk_len_ms)]
            final_chunks.extend(subchunks)
        else:
            final_chunks.append(chunk)

    logger.info(f"Audio split into {len(final_chunks)} chunks (dropped {dropped_count} small chunks).")

    # for i,chunk in enumerate(filtered_chunks):
    #     print(f"Chunk {i}: {len(chunk)/1000}s")

    # Show chunk statistics
    chunk_lengths = [len(chunk) / 1000 for chunk in final_chunks]
    logger.info(
        f"Chunk length stats: min={min(chunk_lengths):.1f}s, max={max(chunk_lengths):.1f}s, avg={sum(chunk_lengths) / len(chunk_lengths):.1f}s")

    return final_chunks


def save_chunks_to_files(audio_chunks, temp_dir):
    """Save audio chunks to temporary files and return file paths."""
    chunk_files = []

    logger.info("Saving chunks to temporary files...")
    for i, chunk in enumerate(tqdm(audio_chunks, desc="Saving chunks")):
        chunk_file = os.path.join(temp_dir, f"chunk_{i:04d}.wav")
        chunk.export(chunk_file, format="wav")
        chunk_files.append(chunk_file)

    return chunk_files


def transcribe_batch(asr_pipeline, chunk_files, generate_kwargs, batch_size=4):
    """
    Transcribe audio files in batches for improved speed.

    Args:
        asr_pipeline: The ASR pipeline
        chunk_files: List of file paths to transcribe
        generate_kwargs: Generation arguments
        batch_size: Number of files to process in each batch

    Returns:
        list: List of transcription results
    """
    results = []

    # Process in batches
    for i in tqdm(range(0, len(chunk_files), batch_size), desc="Transcribing batches"):
        batch_files = chunk_files[i:i + batch_size]

        try:
            # Process batch
            batch_results = asr_pipeline(batch_files, generate_kwargs=generate_kwargs)

            # Handle single result vs list of results
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)

            # Log progress
            for j, result in enumerate(batch_results if isinstance(batch_results, list) else [batch_results]):
                chunk_idx = i + j
                logger.trace(f"Chunk {chunk_idx}: {result['text'][:100]}...")
                logger.trace(f"Chunk {chunk_idx}: {len(result['text'])} symbols")
                tqdm.write(f"Chunk {chunk_idx}: {len(result['text'])} symbols")
                with open("text_a.txt", "a") as f:
                    f.write(result['text'])

        except Exception as e:
            logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
            # Fallback to individual processing for this batch
            for file_path in batch_files:
                try:
                    result = asr_pipeline(file_path, generate_kwargs=generate_kwargs)
                    results.append(result)
                except Exception as e2:
                    logger.error(f"Error processing individual file {file_path}: {e2}")
                    # Add empty result to maintain alignment
                    results.append({"text": "", "chunks": []})

    return results


def merge_transcriptions(results):
    """
    Merge transcription results into a single text.
    Since we split on silence, no overlap handling is needed.
    """
    full_transcript_words = []

    for result in results:
        # Extract words from chunks
        full_transcript_words.append(result['text'])

    # Join all words
    final_text = " ".join(full_transcript_words)

    # Clean up spacing around punctuation
    final_text = final_text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
    final_text = final_text.replace(" :", ":").replace(" ;", ";").strip()

    return final_text


def transcribe_audio(audio_file_path, batch_size=4, language="russian",
                     min_silence_len=700, silence_thresh=-45, max_chunk_len_s=29,
                     model_id="openai/whisper-large-v3", cleanup=True):
    """
    Main function to transcribe audio with batch processing.

    Args:
        audio_file_path (str): Path to the audio file
        batch_size (int): Number of chunks to process in each batch
        language (str): Language for transcription
        min_silence_len (int): Minimum silence length in ms to split on
        silence_thresh (int): Silence threshold in dBFS
        max_chunk_len_s (int): Maximum length of a chunk in seconds
        model_id (str): Whisper model ID
        cleanup (bool): Whether to clean up temporary files

    Returns:
        str: The complete transcription
    """
    # Setup
    asr_pipeline, generate_kwargs = setup_pipeline(model_id, language)

    # Create temporary directory
    temp_dir = "temp_audio_chunks"
    Path(temp_dir).mkdir(exist_ok=True)

    try:
        # Get audio chunks
        audio_chunks = get_audio_chunks(
            audio_file_path,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            max_chunk_len_s=max_chunk_len_s
        )

        # Save chunks to files
        chunk_files = save_chunks_to_files(audio_chunks, temp_dir)

        # Transcribe in batches
        results = transcribe_batch(asr_pipeline, chunk_files, generate_kwargs, batch_size) # list with dicts

        # Merge results
        final_text = merge_transcriptions(results)


        return final_text

    finally:
        # Cleanup
        if cleanup and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Transcribe audio with batch processing")
    # parser.add_argument("audio_file", help="Path to the audio file")
    # parser.add_argument("--batch-size", type=int, default=4,
    #                     help="Number of chunks to process in each batch (default: 4)")
    # parser.add_argument("--language", default="russian",
    #                     help="Language for transcription (default: russian)")
    # parser.add_argument("--min-silence-len", type=int, default=700,
    #                     help="Minimum silence length in ms (default: 700)")
    # parser.add_argument("--silence-thresh", type=int, default=-45,
    #                     help="Silence threshold in dBFS (default: -45)")
    # parser.add_argument("--max-chunk-len", type=int, default=29,
    #                     help="Maximum chunk length in seconds (default: 29)")
    # parser.add_argument("--model", default="openai/whisper-large-v3",
    #                     help="Whisper model ID (default: openai/whisper-large-v3)")
    # parser.add_argument("--no-cleanup", action="store_true",
    #                     help="Don't clean up temporary files")
    # parser.add_argument("--output", "-o", help="Output file path (optional)")

    # args = parser.parse_args()
    
    audio_file="audio_files/lecture2.m4a"
    batch_size=1
    language="russian"
    min_silence_len=700
    silence_thresh=-35
    max_chunk_len=29
    model="openai/whisper-large-v3"
    no_cleanup=True
    output=audio_file.split('/')[-1]+'.txt'
    

    log_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}'
    logger.remove()
    logger.add(sys.stdout, level=9, format=log_format)
    logger.add("log.txt", level=1, format=log_format)

    # Check if input file exists
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        sys.exit(1)

    logger.info(f"Starting transcription of: {audio_file}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Language: {language}")

    # Transcribe
    final_text = transcribe_audio(
        audio_file,
        batch_size=batch_size,
        language=language,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        max_chunk_len_s=max_chunk_len,
        model_id=model,
        cleanup=not no_cleanup
    )

    # Output results
    print("\n--- Full Transcription ---")
    print(final_text)

    # Save to file if requested
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(final_text)
        logger.info(f"Transcription saved to: {output}")








# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", task="transcribe") # optional language
#
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
# forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="ru", task="transcribe")



# original_generate = model.generate
#
# def generate_with_progress(*args, **kwargs):
#     result = original_generate(*args, **kwargs)
#     progress_bar.update(1)  # Update progress bar after each chunk
#     return result
#
# model.generate = generate_with_progress


# asr_pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     feature_extractor=feature_extractor,
#     tokenizer=tokenizer,
#     chunk_length_s=proc_time,
#     device=device,
#     # stride_length_s=(4, 2)
# )


# progress_bar = tqdm(total=int(len(y)/sample_length), desc="Transcribing chunks")
#
# prediction=asr_pipe(sample,generate_kwargs={"forced_decoder_ids": forced_decoder_ids})["text"]
# # prediction=asr_pipe(sample)["text"]
#
# print(prediction)

