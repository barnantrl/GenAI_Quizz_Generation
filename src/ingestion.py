import os
import shutil
import traceback
import imageio_ffmpeg
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

try:
    ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dst = os.path.join(os.getcwd(), "ffmpeg.exe")
    
    if not os.path.exists(ffmpeg_dst):
        shutil.copy(ffmpeg_src, ffmpeg_dst)
    
    os.environ["PATH"] += os.pathsep + os.getcwd()
    
except ImportError:
    print("ERREUR : imageio-ffmpeg n'est pas installé.")

try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        VideoFileClip = None

try:
    import whisper
except ImportError:
    whisper = None

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            raise ValueError(f"Erreur PDF : {e}")

    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    elif ext == ".mp4":
        if VideoFileClip is None: raise ImportError("Manque moviepy")
        if whisper is None: raise ImportError("Manque whisper")
        
        if not os.path.exists("ffmpeg.exe") and shutil.which("ffmpeg") is None:
             raise ImportError("CRITIQUE : ffmpeg.exe est introuvable.")

        try:
            video = None
            try:
                video = VideoFileClip(file_path)
                audio_path = "temp_audio.mp3"
                video.audio.write_audiofile(audio_path, logger=None, codec='mp3')
            except Exception as e:
                raise ValueError(f"Erreur Extraction Audio : {e}")
            finally:
                if video: video.close()

            try:
                model = whisper.load_model("base")
                result = model.transcribe(audio_path, fp16=False)
                transcribed_text = result["text"]
            except Exception as e:
                raise ValueError(f"Erreur Transcription : {e}")
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            return [Document(page_content=transcribed_text, metadata={"source": file_path, "type": "video"})]

        except Exception as e:
            traceback.print_exc()
            raise e
            
    else:
        raise ValueError(f"Format {ext} non supporté")

def create_vector_store(file_path):
    docs = load_document(file_path)
    if not docs: return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.from_documents(chunks, embeddings)