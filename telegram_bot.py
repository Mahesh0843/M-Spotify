import os
import logging
import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from flask import Flask
from threading import Thread

load_dotenv()

# ---------- Render Port Binding Hack ----------
# Render expects a web server to listen on a port. We start one in the background.
web_app = Flask('')

@web_app.route('/')
def home():
    return "Bot is running!"

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    web_app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run_web_server)
    t.daemon = True # Ensures thread dies when main process exits
    t.start()

# ---------- Configuration ----------
CHROMA_PATH = "./chroma_db"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ---------- Setup logging ----------
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Load ChromaDB & Embedder ----------
embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    task_type="retrieval_document"
)

# Initialize Chroma with safety check
if not os.path.exists(CHROMA_PATH):
    logger.error(f"Directory {CHROMA_PATH} not found! Ensure your DB is uploaded.")

client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

try:
    collection = client.get_collection("songs")
    logger.info(f"Loaded collection with {collection.count()} songs.")
except Exception as e:
    logger.error(f"Failed to load collection: {e}")
    # Don't exit(1) immediately if you want to debug on Render logs
    collection = None

# ---------- Helper: search ----------
def search_songs(mood_query, top_k=10):
    if not collection:
        return []
    query_vec = embedder.embed_query(mood_query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    songs = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        score = 1 - results["distances"][0][i]
        songs.append({
            "name": meta.get("Track Name", "Unknown"),
            "artist": meta.get("Artist Name(s)", "Unknown"),
            "url": f"https://open.spotify.com/track/{meta.get('Track URI', '').split(':')[-1]}",
            "score": score
        })
    return songs

# ---------- Bot handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎵 Hi! Send me a mood and I'll find songs for you.")

async def handle_mood(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    try:
        songs = search_songs(user_query, top_k=5)
        if not songs:
            await update.message.reply_text("😕 No matching songs found.")
            return

        response = f"🎵 *Recommendations for:* _{user_query}_\n\n"
        for i, s in enumerate(songs, 1):
            response += f"{i}. [{s['name']} - {s['artist']}]({s['url']}) (Score: {s['score']:.2f})\n"

        await update.message.reply_text(response, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("⚠️ Something went wrong.")

# ---------- Main ----------
def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN missing!")
        return

    # 1. Start the fake web server for Render
    print("🌐 Starting background web server for Render port binding...")
    keep_alive()

    # 2. Start the Telegram Bot
    print("🤖 Starting Telegram Polling...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_mood))
    
    app.run_polling()

if __name__ == "__main__":
    main()