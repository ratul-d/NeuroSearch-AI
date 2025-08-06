from dotenv import load_dotenv
import tempfile,os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", tempfile.gettempdir())
uploaded_files = {}

os.makedirs(UPLOAD_DIR, exist_ok=True)
