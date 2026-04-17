import os
import logging
import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

# ---------- Configuration ----------
CHROMA_PATH = "./chroma_db"
LYRICA_URL = "https://test-0k.onrender.com"   # optional, only if you want to show mood analysis
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")   # you must set this in .env

# ---------- Setup logging ----------
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# # ---------- Load existing ChromaDB ----------
# embedder = GoogleGenerativeAIEmbeddings(
#     model="text-embedding-001",
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )
# Update the model string to include the 'models/' prefix and the correct ID
embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    task_type="retrieval_document" # Highly recommended for your song/lyrics CSV RAG
)

print("✅ Embedder ready with Gemini-001")

client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
try:
    collection = client.get_collection("songs")
    logger.info(f"Loaded collection with {collection.count()} songs.")
except Exception as e:
    logger.error(f"Failed to load collection: {e}")
    exit(1)

# ---------- Helper: search ----------
def search_songs(mood_query, top_k=10):
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
            "name": meta["Track Name"],
            "artist": meta["Artist Name(s)"],
            "url": f"https://open.spotify.com/track/{meta['Track URI'].split(':')[-1]}",  # convert spotify:track:ID to URL
            "score": score
        })
    return songs

# ---------- Bot handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎵 Hi! I'm Spotify Bot.\n\n"
        "Send me a mood (e.g., 'energetic workout songs' or 'sad romantic melodies')\n"
        "and I'll recommend songs from your Spotify playlists."
    )

async def handle_mood(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    logger.info(f"User {update.effective_user.id} asked: {user_query}")

    await update.message.reply_text(f"🎧 Searching for: *{user_query}*...", parse_mode="Markdown")

    try:
        songs = search_songs(user_query, top_k=10)
        if not songs:
            await update.message.reply_text("😕 No matching songs found. Try a different mood.")
            return

        # Build response message
        response = f"🎵 *Top recommendations for:* _{user_query}_\n\n"
        for i, s in enumerate(songs[:5], 1):
            response += f"{i}. [{s['name']} - {s['artist']}]({s['url']})  (score: {s['score']:.2f})\n"
        if len(songs) > 5:
            response += f"\n_...and {len(songs)-5} more. Open the links to play on Spotify._"

        await update.message.reply_text(response, parse_mode="Markdown", disable_web_page_preview=False)
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("⚠️ Something went wrong. Please try again later.")

# ---------- Main ----------
def main():
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not found in .env file.")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_mood))

    print("🤖 Bot is running. Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()