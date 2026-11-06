#!/usr/bin/env python3
"""Morgen-Podcast

Ablauf:
1. Nachrichten per RSS holen (Tagesschau, Spiegel, ZDF, SZ, Sächsische Zeitung)
2. Gemini schreibt (a) einen Nachrichtenblock im Radio-Stil und (b) ein Streitgespräch Alex/Sam
3. Gemini-Stimmen vertonen beides: Alex liest die Nachrichten allein, danach reden Alex und Sam
4. MP3 + Podcast-Feed (docs/feed.xml) + kleine Webseite (docs/index.html) werden gebaut

Aufruf:
  python podcast.py               # komplette Folge für heute (überspringt, wenn schon vorhanden)
  python podcast.py --force       # heutige Folge neu erzeugen
  python podcast.py --dry-run     # nur Nachrichten holen und zählen (kein API-Zugriff nötig)
  python podcast.py --text-only   # nur Texte erzeugen (vorschau.txt), keine Stimmen
  python podcast.py --rebuild-feed
"""
from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import html
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Berlin")
ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
EPISODES = DOCS / "episodes"
SAMPLE_RATE = 24000  # Gemini-TTS liefert 24 kHz, 16 Bit, mono


def env(name: str, default: str) -> str:
    """Umgebungsvariable lesen; leere Werte zählen als nicht gesetzt."""
    return os.environ.get(name) or default


# Modelle und Stimmen lassen sich über Variablen ändern, ohne den Code anzufassen.
TEXT_MODEL = env("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
TTS_MODEL = env("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
VOICE_ALEX = env("VOICE_ALEX", "Charon")
VOICE_SAM = env("VOICE_SAM", "Kore")
KEEP_EPISODES = int(env("KEEP_EPISODES", "14"))
TTS_CHUNK_CHARS = int(env("TTS_CHUNK_CHARS", "3500"))
BASE_URL = env("PODCAST_BASE_URL", "http://localhost:8000").rstrip("/")

PODCAST_TITLE = "Morgen-Podcast"
PODCAST_DESCRIPTION = (
    "Täglicher KI-generierter Nachrichtenpodcast: erst ein Nachrichtenblock im Radio-Stil, "
    "danach diskutieren Alex und Sam die Themen mit Fakten, Gegenargumenten und Lösungsansätzen."
)

# Die Adressen sind Standard-RSS-Feeds. Fällt einer aus, wird er übersprungen (siehe Log).
FEEDS: dict[str, list[tuple[str, str]]] = {
    "GEOPOLITIK": [
        ("Tagesschau", "https://www.tagesschau.de/ausland/index~rss2.xml"),
        ("Spiegel", "https://www.spiegel.de/ausland/index.rss"),
        ("ZDFheute", "https://www.zdf.de/rss/zdf/nachrichten"),
    ],
    "INNENPOLITIK": [
        ("Tagesschau", "https://www.tagesschau.de/inland/index~rss2.xml"),
        ("Spiegel", "https://www.spiegel.de/politik/index.rss"),
        ("Süddeutsche Zeitung", "https://rss.sueddeutsche.de/rss/Politik"),
    ],
    "DRESDEN (lokal)": [
        ("Sächsische Zeitung", "https://www.saechsische.de/rss/dresden"),
    ],
    "WISSENSCHAFT UND TECHNIK": [
        ("Spiegel", "https://www.spiegel.de/wissenschaft/index.rss"),
        ("Süddeutsche Zeitung", "https://rss.sueddeutsche.de/rss/Wissen"),
    ],
}

WEEKDAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
MONTHS = [
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]


def german_date(d: dt.date) -> str:
    return f"{WEEKDAYS[d.weekday()]}, {d.day}. {MONTHS[d.month - 1]} {d.year}"


# --------------------------------------------------------------------------- Nachrichten holen


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def fetch_category(name: str, feeds: list[tuple[str, str]], per_feed: int = 10) -> list[dict]:
    import feedparser
    import requests

    items: list[dict] = []
    seen: set[str] = set()
    for source, url in feeds:
        try:
            r = requests.get(url, headers={"User-Agent": "morgen-podcast/1.0"}, timeout=25)
            r.raise_for_status()
            entries = feedparser.parse(r.content).entries[:per_feed]
        except Exception as e:  # ein defekter Feed darf nicht alles stoppen
            print(f"WARNUNG: {source} ({name}) nicht lesbar: {str(e)[:150]}", file=sys.stderr)
            continue
        count = 0
        for entry in entries:
            title = strip_html(entry.get("title", ""))
            if not title or title.lower() in seen:
                continue
            seen.add(title.lower())
            summary = strip_html(entry.get("summary", ""))[:350]
            items.append({"source": source, "title": title, "summary": summary})
            count += 1
        print(f"{name}: {source}: {count} Meldungen")
    return items


def material_text(by_cat: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    for cat, items in by_cat.items():
        lines.append(f"=== {cat} ===")
        for i in items:
            tail = f" – {i['summary']}" if i["summary"] else ""
            lines.append(f"[{i['source']}] {i['title']}{tail}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- Texte erzeugen


def with_retry(fn, tries: int = 5):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            fatal = any(k in msg for k in ("API_KEY_INVALID", "PERMISSION_DENIED", "API key not valid"))
            if fatal or attempt == tries - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f"Fehler: {msg[:200]} – neuer Versuch in {wait}s", file=sys.stderr)
            time.sleep(wait)


def get_client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY fehlt (als Secret im Repository oder als Umgebungsvariable setzen).")
    return genai.Client(api_key=key)


def generate_text(client, prompt: str) -> str:
    from google.genai import types

    def call():
        resp = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.6),
        )
        if not resp.text:
            raise RuntimeError("Leere Antwort vom Textmodell")
        return resp.text

    return with_retry(call)


def clean(text: str) -> str:
    """Markdown-Reste entfernen, damit nichts davon vorgelesen wird."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"[*#_`>]+", "", text)
    return text.strip()


NEWS_PROMPT = """Du bist Nachrichtenredakteur und schreibst das Manuskript für eine ausführliche Radio-Nachrichtensendung auf Deutsch. Heute ist {datum}.

Regeln:
- Verwende ausschließlich Informationen aus dem Material unten. Erfinde nichts. Fehlen Details, lass sie weg, statt zu raten.
- Struktur in dieser Reihenfolge: bis zu 6 Meldungen Geopolitik (inklusive internationaler Themen), bis zu 6 Meldungen Innenpolitik und Lokales (mindestens 2 aus Dresden, wenn vorhanden), bis zu 4 Meldungen Wissenschaft und Technik, dazu 1 skurriles Naturwunder oder kurioses Wissenschaftsthema. Gibt das Material weniger her, nimm weniger.
- Jede Meldung so ausführlich, wie das Material es hergibt, höchstens 6 Sätze: was ist passiert, wer ist beteiligt, Zahlen, Ort, Hintergrund. Sachlich, ohne Wertung, ohne Füllsätze. Nenne die Quelle kurz ("meldet die Tagesschau").
- Beginne mit: "Guten Morgen. Heute ist {datum}. Das sind die Nachrichten." Kurze Überleitungen zwischen den Blöcken, zum Beispiel "Nun zur Innenpolitik." Ende mit: "Das waren die Nachrichten. Jetzt sprechen wir darüber."
- Reiner Fließtext in Absätzen, eine Meldung pro Absatz. Keine Markdown-Zeichen, keine Listen, keine Überschriften, keine Sprecherlabels.
- Schreibe Zahlen und Abkürzungen so, dass man sie gut vorlesen kann.

MATERIAL:
{material}
"""

DEBATE_PROMPT = """Schreibe das Skript für ein Gespräch zwischen den zwei Podcast-Hosts Alex und Sam auf Deutsch. Sie haben gerade die folgenden Nachrichten gehört und diskutieren sie jetzt.

Regeln:
- Ton: offen und freundlich, aber durchaus kontrovers. Die beiden vertreten bei den meisten Themen unterschiedliche Positionen und versuchen, einander mit Fakten zu überzeugen. Argument, Gegenargument, am Ende ein Fazit oder ein offener Punkt.
- Nutze nur Fakten aus den Nachrichten und gesichertes Allgemeinwissen. Was sich daraus nicht belegen lässt, sagen die Hosts ausdrücklich ("das lässt sich aus den Meldungen nicht sagen"). Keine erfundenen Zahlen, Zitate oder Studien.
- Faktencheck: Wo Behauptungen oder Mythen im Umlauf sind, prüfen sie diese sachlich. Rabbit-Hole: Bei einigen Themen gibt es kurz einen überraschenden Hintergrund.
- Bei gesellschaftlichen Themen schauen sie auf Arm gegen Reich: Welche Belastungen und Ängste entstehen für wen? Welche Lösungsansätze gibt es, und was würden sie für die jeweilige Gruppe bedeuten?
- Gehe auf alle Themen ein; kleinere kürzer, gesellschaftlich wichtige ausführlicher. Insgesamt etwa 1800 bis 2500 Wörter.
- Sam beginnt mit einer kurzen Begrüßung, Alex verabschiedet am Ende.
- Format streng: jede Zeile beginnt mit "Alex:" oder "Sam:". Keine anderen Zeilen, keine Regieanweisungen, keine Markdown-Zeichen, keine Aufzählungen.

NACHRICHTEN:
{news}
"""

DIALOGUE_LINE = re.compile(r"^\s*(Alex|Sam)\s*:\s*(.+?)\s*$")


def parse_dialogue(text: str) -> list[str]:
    turns = []
    for line in clean(text).splitlines():
        m = DIALOGUE_LINE.match(line)
        if m:
            turns.append(f"{m.group(1)}: {m.group(2)}")
    if len(turns) < 6:
        raise RuntimeError("Das Gesprächsskript hat zu wenige Zeilen im Format 'Alex:' / 'Sam:'.")
    return turns


# --------------------------------------------------------------------------- Vertonen


def chunk_lines(text: str, limit: int) -> list[str]:
    """Teilt an Absatz-/Zeilengrenzen in Stücke bis ca. limit Zeichen."""
    parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > limit:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)


def synthesize(client, prompt: str, speakers: dict[str, str]) -> bytes:
    """Ein Textstück vertonen. speakers: {Name: Stimme}; zwei Einträge = Gespräch."""
    from google.genai import types

    def voice(name: str):
        return types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=name))

    if len(speakers) == 1:
        speech = types.SpeechConfig(voice_config=voice(next(iter(speakers.values()))))
    else:
        speech = types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(speaker=who, voice_config=voice(v))
                    for who, v in speakers.items()
                ]
            )
        )

    def call():
        resp = client.models.generate_content(
            model=TTS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=speech),
        )
        for cand in resp.candidates or []:
            for part in (cand.content.parts if cand.content else []) or []:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data
        raise RuntimeError("Keine Audiodaten in der Antwort")

    data = with_retry(call)
    time.sleep(3)  # etwas Abstand wegen Mindestabständen der API
    return data


def pcm_to_mp3(pcm: bytes, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
            "-codec:a", "libmp3lame", "-b:a", "64k", str(out),
        ],
        input=pcm,
        check=True,
    )


def render_audio(client, news: str, turns: list[str]) -> bytes:
    pcm = b""
    news_chunks = chunk_lines(news, TTS_CHUNK_CHARS)
    talk_chunks = chunk_lines("\n".join(turns), TTS_CHUNK_CHARS)
    total = len(news_chunks) + len(talk_chunks)
    n = 0

    for chunk in news_chunks:  # Teil 1: Alex allein
        n += 1
        print(f"Vertone Nachrichtenblock {n}/{total} ...")
        prompt = (
            "Lies den folgenden Text auf Deutsch wie ein erfahrener Radio-Nachrichtensprecher vor: "
            "ruhig, klar und seriös. Füge nichts hinzu und lass nichts weg:\n\n" + chunk
        )
        pcm += synthesize(client, prompt, {"Alex": VOICE_ALEX}) + silence(0.5)

    pcm += silence(1.5)

    for chunk in talk_chunks:  # Teil 2: Alex und Sam
        n += 1
        print(f"Vertone Gespräch {n}/{total} ...")
        prompt = (
            "Lies das folgende Gespräch zwischen Alex und Sam auf Deutsch vor, freundlich, "
            "lebendig und natürlich:\n\n" + chunk
        )
        pcm += synthesize(client, prompt, {"Alex": VOICE_ALEX, "Sam": VOICE_SAM}) + silence(0.4)
    return pcm


# --------------------------------------------------------------------------- Feed und Webseite


def hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def load_metas() -> list[dict]:
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in EPISODES.glob("*.json")]
    return sorted(metas, key=lambda m: m["date"], reverse=True)


def build_feed() -> None:
    metas = load_metas()
    items = []
    for m in metas:
        url = f"{BASE_URL}/episodes/{m['date']}.mp3"
        pub = email.utils.format_datetime(dt.datetime.fromisoformat(m["published"]))
        items.append(
            f"""    <item>
      <title>{escape(m['title'])}</title>
      <description>{escape(m['description'])}</description>
      <pubDate>{pub}</pubDate>
      <guid isPermaLink="false">morgen-podcast-{m['date']}</guid>
      <enclosure url="{escape(url)}" length="{m['bytes']}" type="audio/mpeg"/>
      <itunes:duration>{hms(m['seconds'])}</itunes:duration>
    </item>"""
        )
    feed = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>{escape(PODCAST_TITLE)}</title>
    <link>{escape(BASE_URL)}</link>
    <description>{escape(PODCAST_DESCRIPTION)}</description>
    <language>de-de</language>
    <itunes:author>{escape(PODCAST_TITLE)}</itunes:author>
    <itunes:explicit>false</itunes:explicit>
{chr(10).join(items)}
  </channel>
</rss>
"""
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "feed.xml").write_text(feed, encoding="utf-8")

    rows = "\n".join(
        f"""<section><h2>{escape(m['title'])}</h2>
<audio controls preload="none" src="episodes/{m['date']}.mp3"></audio>
<p><a href="episodes/{m['date']}.txt">Text zur Folge</a></p></section>"""
        for m in metas
    )
    page = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(PODCAST_TITLE)}</title>
<style>
:root{{color-scheme:light dark}}
body{{font-family:system-ui,sans-serif;max-width:40rem;margin:0 auto;padding:1rem 16px}}
audio{{width:100%}} section{{margin:1.5rem 0}} h2{{font-size:1.1rem;margin:0 0 .5rem}}
</style></head><body>
<h1>{escape(PODCAST_TITLE)}</h1>
<p>{escape(PODCAST_DESCRIPTION)}</p>
<p>Feed für die Podcast-App: <a href="feed.xml">feed.xml</a></p>
{rows}
</body></html>
"""
    (DOCS / "index.html").write_text(page, encoding="utf-8")


def prune_old() -> None:
    for m in load_metas()[KEEP_EPISODES:]:
        for ext in ("mp3", "txt", "json"):
            (EPISODES / f"{m['date']}.{ext}").unlink(missing_ok=True)


def set_output(generated: bool) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"generated={'true' if generated else 'false'}\n")


# --------------------------------------------------------------------------- Hauptablauf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="heutige Folge neu erzeugen")
    ap.add_argument("--dry-run", action="store_true", help="nur Nachrichten holen")
    ap.add_argument("--text-only", action="store_true", help="nur Texte erzeugen, keine Stimmen")
    ap.add_argument("--rebuild-feed", action="store_true", help="nur Feed und Webseite neu bauen")
    args = ap.parse_args()

    if args.rebuild_feed:
        build_feed()
        print("Feed neu gebaut.")
        return 0

    now = dt.datetime.now(TZ)
    day = now.date().isoformat()
    mp3 = EPISODES / f"{day}.mp3"
    preview_only = args.dry_run or args.text_only
    if mp3.exists() and not args.force and not preview_only:
        print(f"Folge für {day} existiert bereits – nichts zu tun.")
        set_output(False)
        return 0

    by_cat = {cat: fetch_category(cat, feeds) for cat, feeds in FEEDS.items()}
    total = sum(len(v) for v in by_cat.values())
    if total == 0:
        sys.exit("Keine Nachrichten gefunden – alle Feeds nicht erreichbar.")
    if args.dry_run:
        print(f"Dry-Run fertig: {total} Meldungen gesammelt.")
        return 0

    material = material_text(by_cat)
    client = get_client()
    datum = german_date(now.date())

    print("Schreibe Nachrichtenblock ...")
    news = clean(generate_text(client, NEWS_PROMPT.format(datum=datum, material=material)))
    print("Schreibe Gespräch ...")
    turns = parse_dialogue(generate_text(client, DEBATE_PROMPT.format(news=news)))
    script = news + "\n\n--- GESPRÄCH ---\n\n" + "\n".join(turns) + "\n"

    if args.text_only:
        (ROOT / "vorschau.txt").write_text(script, encoding="utf-8")
        print("Texte in vorschau.txt gespeichert.")
        return 0

    pcm = render_audio(client, news, turns)
    pcm_to_mp3(pcm, mp3)
    (EPISODES / f"{day}.txt").write_text(script, encoding="utf-8")
    meta = {
        "date": day,
        "title": f"Morgen-Podcast {datum}",
        "description": (
            f"Nachrichtenblock und Diskussion vom {datum}. KI-generiert aus Meldungen von "
            "Tagesschau, Spiegel, ZDF, Süddeutscher Zeitung und Sächsischer Zeitung."
        ),
        "published": now.isoformat(),
        "bytes": mp3.stat().st_size,
        "seconds": len(pcm) // (SAMPLE_RATE * 2),
    }
    (EPISODES / f"{day}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    prune_old()
    build_feed()
    set_output(True)
    print(f"Fertig: {mp3.name}, {hms(meta['seconds'])}, {meta['bytes'] // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
