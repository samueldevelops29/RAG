# app.py
import uuid
from fastapi import FastAPI, HTTPException, Request, Form, Response, UploadFile, File, APIRouter, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from feedgen.feed import FeedGenerator
from datetime import datetime, timezone
from pydantic import BaseModel
from openai_connect import generate_skill_tree_from_documents, save_skill_tree_to_file, generate_15_questions_from_skill_tree
from vector_database import query_qdrant, load_documents, client as qdrant_client, model as sentence_model, COLLECTION_NAME, PointStruct, text_splitter
import json
import os
import shutil
import tempfile
import numpy as np
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
router = APIRouter()

client = OpenAI(api_key="API_KEY")  # In Umgebungsvariable setzen, env

class Antwort(BaseModel):
    frage: str
    richtige_antwort: str
    gegebene_antwort: str
    korrekt: bool
    quelle: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def verarbeite_query(request: Request):
    body = await request.json()
    user_input = body.get("user_input")

    if not user_input:
        return JSONResponse(status_code=400, content={"error": "Kein user_input übergeben."})

    dokumente = query_qdrant(user_input)
    kontext = "\n\n".join([doc["text"] for doc in dokumente]) if dokumente else "❌ Kein Kontext gefunden."

    try:
        gpt_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Beantworte die Frage des Nutzers so hilfreich und konkret wie möglich basierend auf dem bereitgestellten Kontext."},
                {"role": "user", "content": f"Kontext:\n{kontext}\n\nFrage:\n{user_input}"}
            ],
            temperature=0.7,
            max_tokens=500
        )

        antwort = gpt_response.choices[0].message.content.strip()
        quellen = [doc.get("quelle", "❌ Keine Quelle") for doc in dokumente]

        return JSONResponse(content={
            "antwort": antwort,
            "quellen": quellen,
            "kontext": kontext
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Fehler bei der LLM-Antwort: {str(e)}"})


@app.get("/fragen")
async def frage_generator():
    try:
        with open("15_selektierte_fragen_LLM.json", "r", encoding="utf-8") as f:
            fragen = json.load(f)
        return JSONResponse(content=fragen)
    except FileNotFoundError:
        return JSONResponse(content={"error": "15_selektierte_fragen_LLM.json nicht gefunden."}, status_code=404)

@app.post("/auswertung")
async def auswertung(request: Request, background_tasks: BackgroundTasks):
    try:
        form_data = await request.form()
        antworten = dict(form_data)

        with open("15_selektierte_fragen_LLM.json", "r", encoding="utf-8") as f:
          fragen_dict = json.load(f)

        fehlerthemen = []

        for skill, antwort_index in antworten.items():
          try:
              frage_info = fragen_dict[skill]
              antworten_liste = frage_info["antworten"]
              index = int(antwort_index)

              if not antworten_liste[index]["korrekt"]:
                richtige = next((a["text"] for a in antworten_liste if a["korrekt"]), "❌ Keine richtige Antwort gefunden")
                chunk_result = query_qdrant(skill)

                fehlerthemen.append({
                    "skill": skill,
                    "frage": frage_info["frage"],
                    "gegebene_antwort": antworten_liste[index]["text"],
                    "richtige_antwort": richtige,
                    "chunk": chunk_result[0]["text"] if chunk_result else "❌ Kein passender Kontext gefunden.",
                    "quelle": frage_info.get("quelle", "❌ Quelle unbekannt")
                })
          except Exception as e:
            print(f"Fehler bei der Auswertung von Skill {skill}: {e}")
            continue
        # ✅ Lernempfehlungen sofort zurückgeben
        return JSONResponse(content={"empfehlungen": fehlerthemen})
    except Exception as e:
     print(f"❌ Fehler bei /auswertung: {e}")
     return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/podcasts")
async def podcasts(request: Request):
    """
    Nimmt Fehlerthemen entgegen und erstellt (falls nötig) Podcasts als MP3.
    """
    try:
        body = await request.json()
        fehlerthemen = body.get("fehlerthemen", [])

        if not fehlerthemen:
            return JSONResponse(status_code=400, content={"error": "Keine fehlerthemen übergeben."})

        print(f"🎧 Starte Podcastgenerierung für {len(fehlerthemen)} Themen...")
        generate_podcasts(fehlerthemen)

        return JSONResponse(content={"message": "🎧 Podcasts werden erstellt."})
    except Exception as e:
        print(f"❌ Fehler bei /podcasts: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})



def generate_podcasts(fehlerthemen: list):
    """
    Erstellt aus Fehlerthemen MP3-Podcasts (Text-to-Speech).
    """
    try:
        os.makedirs("audio", exist_ok=True)
        for thema in fehlerthemen:
            try:
                chunk = thema["chunk"]
                skill_name = thema["skill"]
                # Dateiname sicher machen (keine Leerzeichen, max. 50 Zeichen)
                filename = f"audio/{skill_name[:50].replace(' ', '_')}.mp3"

                # Wenn Datei schon existiert, überspringen
                if os.path.exists(filename):
                    print(f"ℹ️ Datei existiert bereits: {filename}")
                    continue

                frage = thema["frage"]

                # 1️⃣ Erklärungstext vom GPT-Modell erzeugen
                explanation = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Du bist ein KI-Lerncoach für Softwareentwickler. Erkläre die fachliche Frage auf leicht verständliche Weise, indem du das nötige Hintergrundwissen aus dem bereitgestellten Kontext nutzt."},
                        {"role": "user", "content": f"""Frage des Nutzers:
{frage}
Kontext zur Antwort (Hintergrundwissen):
{chunk}
Erkläre dem Nutzer, was die richtige Antwort gewesen wäre und warum. Gehe auch darauf ein, warum diese Frage relevant ist."""}
                    ],
                    temperature=0.7,
                    max_tokens=500
                ).choices[0].message.content.strip()

                # 2️⃣ Erklärung in Audio umwandeln
                audio_response = client.audio.speech.create(
                    model="tts-1",
                    voice="nova",
                    input=explanation
                )

                with open(filename, "wb") as f:
                    f.write(audio_response.content)

                print(f"✅ Podcast erstellt: {filename}")

            except Exception as e:
                print(f"❌ Fehler bei der Podcast-Erstellung für Skill '{thema.get('skill', 'unbekannt')}': {e}")
    except Exception as outer:
        print("🔥 Schwerwiegender Fehler im Podcast-Task:", outer)



# Audio-Dateien bereitstellen
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

@app.get("/rss_feed")
async def rss_feed():
    audio_dir = "audio"
    os.makedirs(audio_dir, exist_ok=True)

    fg = FeedGenerator()
    fg.title("Dein Lern-Podcast")
    fg.link(href="http://127.0.0.1:8000/rss_feed", rel="self")
    fg.description("Automatisch generierte Audioerklärungen zu deinen Fehlern.")

    for filename in os.listdir(audio_dir):
        if filename.endswith(".mp3"):
            title = filename.replace("_", " ").replace(".mp3", "")
            url = f"http://127.0.0.1:8000/audio/{filename}"
            fe = fg.add_entry()
            fe.title(title)
            fe.link(href=url)
            fe.enclosure(url, 0, "audio/mpeg")
            fe.pubDate(datetime.now(timezone.utc))

    fg.rss_file("rss_feed.xml")

    with open("rss_feed.xml", "r", encoding="utf-8") as f:
        content = f.read()

    return Response(content=content, media_type="application/rss+xml", headers={
        "Access-Control-Allow-Origin": "*"
    })


def get_all_documents_from_qdrant(limit=100):
    print(f"Hole bis zu {limit} Dokumente aus Qdrant für den Gesamtkontext...")
    random_vector = np.random.rand(384).tolist()

    response = qdrant_client.scroll(
        collection_name="my_documents",
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    records = response[0]
    return [record.payload for record in records]


@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    try:
        save_path = f"user_uploads/{uuid.uuid4()}_{file.filename}"
        os.makedirs("user_uploads", exist_ok=True)
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())
        print(f"📁 Datei temporär gespeichert unter: {save_path}")

        file_extension = file.filename.split('.')[-1].lower()
        load_documents(custom_file=(file.filename, save_path, file_extension))

        all_documents_from_db = get_all_documents_from_qdrant(limit=200)

        if not all_documents_from_db:
            raise HTTPException(status_code=500, detail="Konnte keine Dokumente aus der Datenbank abrufen, um Skill-Tree zu erstellen.")

        print("🌳 Erzeuge neuen Skill-Tree basierend auf dem Gesamtkontext aus Qdrant...")
        skill_tree = generate_skill_tree_from_documents(all_documents_from_db)
        save_skill_tree_to_file(skill_tree)
        print("✅ Skill-Tree aktualisiert.")

        print("❓ Erzeuge neue Fragen...")
        generate_15_questions_from_skill_tree("skill_trees/skill_tree_auto3.json")
        print("✅ Neue Fragen generiert.")

        os.remove(save_path)

        return {"message": "✅ Datei erfolgreich verarbeitet und zur Datenbank hinzugefügt. Skill-Tree und Fragen wurden aktualisiert."}

    except Exception as e:
        import traceback
        print(f"❌ Schwerwiegender Fehler in /upload_document: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ein interner Fehler ist aufgetreten: {str(e)}")

from fastapi.responses import StreamingResponse

@app.post("/feedback_stream")
async def feedback_stream(request: Request):
    try:
        answer_data = await request.json()

        def stream_generator():
            prompt = f"""
Du bist ein Lerntutor. Analysiere die folgenden Quiz-Antworten und gib dem Nutzer eine freundliche, aber konkrete Rückmeldung zu seinem Wissen.

Empfiehl, welche Quellen er sich nochmal anschauen sollte und warum. Begründe das pädagogisch.

Deine Empfehlung soll kurz und prägnant formuliert werden.

### Antwortdaten:
{json.dumps(answer_data, indent=2, ensure_ascii=False)}
"""

            response = client.chat.completions.create(
                model="gpt-4",
                stream=True,
                messages=[
                    {"role": "system", "content": "Du bist ein hilfsbereiter Lerntutor, der nicht nur Erklärungen und Empfehlungen ausgibt, sondernauch den Verweis zu den selbst generierten Podcasts betont, um die jeweiligen Inhalte eigenständig zu lernen."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )

            for chunk in response:
             delta = chunk.choices[0].delta
             content = getattr(delta, "content", "")
             if content:
              yield content


        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as e:
        print("❌ Fehler im /feedback_stream:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get("/podcast_page", response_class=HTMLResponse)
async def podcast_page():
    """
    Zeigt eine einfache HTML-Seite mit allen Podcasts im audio/-Verzeichnis an.
    """
    audio_dir = "audio"
    os.makedirs(audio_dir, exist_ok=True)

    files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]

    html_content = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>Deine Lern-Podcasts</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f6f8; padding: 20px; color: #222; }
            h1 { color: #0a4a8f; }
            .podcast {
                background: white;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                padding: 15px;
                margin-bottom: 15px;
            }
            audio { width: 100%; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>🎧 Deine Lern-Podcasts</h1>
        <p>Hier kannst du alle Podcasts direkt im Browser anhören:</p>
    """

    if not files:
        html_content += "<p>❌ Noch keine Podcasts vorhanden.</p>"
    else:
        for file in sorted(files):
            title = file.replace("_", " ").replace(".mp3", "")
            url = f"/audio/{file}"
            html_content += f"""
            <div class="podcast">
                <h3>{title}</h3>
                <audio controls>
                    <source src="{url}" type="audio/mpeg">
                    Dein Browser unterstützt kein Audio-Tag.
                </audio>
            </div>
            """

    html_content += """
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)
