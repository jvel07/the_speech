from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="ACCESS_TOKEN_GOES_HERE")

# apply the pipeline to an audio file
diarization = pipeline("audio.wav")

# dump the diarization output to disk using RTTM format
with open("data/audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)