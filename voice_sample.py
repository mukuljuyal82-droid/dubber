"""Make short voice samples to compare voices. Needs: pip install edge-tts"""
import asyncio
import edge_tts

HI = "नमस्ते दोस्तों! आज की कहानी में हम देखेंगे कि कैसे एक साधारण लड़के ने अपनी किस्मत बदल दी।"
EN = "Hello everyone! In today's story, we will see how an ordinary boy changed his destiny."

JOBS = [
    (HI, "hi-IN-SwaraNeural", "hindi_swara.mp3"),
    (HI, "hi-IN-MadhurNeural", "hindi_madhur.mp3"),
    (EN, "en-US-AriaNeural", "english_aria.mp3"),
    (EN, "en-US-GuyNeural", "english_guy.mp3"),
]


async def main():
    for text, voice, name in JOBS:
        await edge_tts.Communicate(text, voice).save(name)
        print("saved", name)

asyncio.run(main())
