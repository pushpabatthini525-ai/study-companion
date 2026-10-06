from youtube_transcript_api import YouTubeTranscriptApi

def get_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url.rstrip("/").split("/")[-1].split("?")[0]

def fetch_transcript(video_id):
    api = YouTubeTranscriptApi()
    try:
        data = api.fetch(video_id, languages=["en", "hi", "te"])
    except Exception:
        # aa languages lo lekapothe, unna edaina transcript teesukuntundi
        data = next(iter(api.list(video_id))).fetch()
    return [{"text": s.text, "start": s.start} for s in data]

def build(vid, start, buf):
    t = int(start)
    return {
        "text": " ".join(buf),
        "source": f"https://youtu.be/{vid}?t={t}",
        "timestamp": f"{t//60}:{t%60:02d}",
    }

def make_chunks(url, window=60):
    vid = get_video_id(url)
    lines = fetch_transcript(vid)
    chunks, buf, start = [], [], 0
    for l in lines:
        if not buf:
            start = l["start"]
        buf.append(l["text"])
        if l["start"] - start >= window:
            chunks.append(build(vid, start, buf))
            buf = []
    if buf:
        chunks.append(build(vid, start, buf))
    return chunks

if __name__ == "__main__":
    url = input("YouTube link: ")
    chunks = make_chunks(url)
    print(len(chunks), "chunks")
    print(chunks[0])