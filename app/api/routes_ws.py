import traceback, asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.pdf_utils import extract_pdf_chunks, download_pdf
from app.services.embedding import build_faiss_index, search_faiss
from app.services.qa import ask_llama_3_70b
from app.services.embedding import EmbeddingModel
from app.config import uploaded_files,UPLOAD_DIR

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Check if a PDF link was provided as a query parameter
        pdf_url = websocket.query_params.get("pdfLink")
        if pdf_url:
            await websocket.send_text("Downloading PDF from remote source...Please Wait....")
            pdf_path = await asyncio.to_thread(download_pdf, pdf_url)
        else:
            await websocket.send_text("Error! Refresh Page")

        # Process the PDF and send progress messages
        await websocket.send_text("Extracting and processing the PDF...Please Wait....")
        text_chunks = await asyncio.to_thread(extract_pdf_chunks, pdf_path)
        await websocket.send_text("Loading the embedding model...Please Wait....")
        embed_model = await asyncio.to_thread(EmbeddingModel)
        await websocket.send_text("Building FAISS index...Please Wait....")
        faiss_index, stored_chunks = build_faiss_index(text_chunks, embed_model)
        await websocket.send_text("Ready for Q&A!\n🔹 Ask a question about the paper:")

        chat_history = []

        # Process incoming queries from the client
        while True:
            query = await websocket.receive_text()
            await websocket.send_text("Processing your query...")

            full_context = "\n".join(
                [f"User: {q}\nBot: {a}" for q, a in chat_history[-3:]]
            )

            relevant_chunks = search_faiss(query, faiss_index, stored_chunks, embed_model)

            combined_context = full_context + "\n" + "\n".join(relevant_chunks)

            answer = ask_llama_3_70b(query, [combined_context])

            chat_history.append((query, answer))
            if len(chat_history) > 3:
                chat_history = chat_history[-3:]

            result_message = "Answer:\n" + answer
            await websocket.send_text(result_message)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        tb = traceback.format_exc()
        await websocket.send_text(f"An error occurred: {str(e)}")
        await websocket.send_text(f"An error occurred:\n{tb}")
        await websocket.close()

@router.websocket("/wsu")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for processing chat queries. The client must pass the file id as a query parameter (?file=).
    """
    await websocket.accept()
    file_id = websocket.query_params.get("file")
    if not file_id or file_id not in uploaded_files:
        await websocket.send_text("Error! No valid PDF file provided. Refresh page and upload a PDF.")
        await websocket.close()
        return

    pdf_path = uploaded_files[file_id]
    try:
        # Send progress messages to the client
        await websocket.send_text("Processing your uploaded PDF...please wait.")
        text_chunks = await asyncio.to_thread(extract_pdf_chunks, pdf_path)
        await websocket.send_text("Loading embedding model...please wait.")
        embed_model = await asyncio.to_thread(EmbeddingModel)  # Uses OPENAI_API_KEY from env
        await websocket.send_text("Building FAISS index...please wait.")
        faiss_index, stored_chunks = build_faiss_index(text_chunks, embed_model)
        await websocket.send_text("Ready for Q&A!\n🔹 Ask a question about the paper:")

        wsu_chat_history=[]
        # Process incoming queries
        while True:
            query = await websocket.receive_text()
            await websocket.send_text("Processing your query...")

            full_context = "\n".join(
                [f"User: {q}\nBot: {a}" for q, a in wsu_chat_history[-3:]]
            )

            relevant_chunks = search_faiss(query, faiss_index, stored_chunks, embed_model)

            combined_context = full_context + "\n" + "\n".join(relevant_chunks)

            answer = ask_llama_3_70b(query, [combined_context])

            wsu_chat_history.append((query, answer))
            if len(wsu_chat_history) > 3:
                wsu_chat_history = wsu_chat_history[-3:]

            result_message = "Answer:\n" + answer
            await websocket.send_text(result_message)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        tb = traceback.format_exc()
        await websocket.send_text(f"An error occurred: {str(e)}")
        await websocket.send_text(f"Details:\n{tb}")
        await websocket.close()