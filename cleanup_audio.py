# -*- coding: utf-8 -*-
"""
Script to clean up old audio files, keeping only output_en.mp3 and output_hi.mp3
"""
import os
from modules.text_to_speech import cleanup_old_audio_files

if __name__ == "__main__":
    print("Cleaning up old audio files...")
    removed = cleanup_old_audio_files()
    print(f"Cleanup complete! Removed {removed} old audio files.")
    print("Only output_en.mp3 and output_hi.mp3 remain.")
