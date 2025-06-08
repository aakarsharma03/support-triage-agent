# --- app.py ---
# This file will now only contain the FastAPI app and embedding logic.
import modal
from fastapi import FastAPI, HTTPException, Request

app = modal.App("support-triage-agent")
image = modal.Image.debian_slim().pip_install_from_requirements("./requirements.txt")

api = FastAPI()

@app.function(image=image)
def draft_reply(category, priority, ticket):
    return f"""
Hello! We've received your {category} ticket and marked it as {priority} priority. Could you please provide any additional details that might help us resolve your issue?\n\nYour message: "{ticket}"\n\nBest regards,\nSupport Team
"""

# Example tickets for robust categorization
EXAMPLE_TICKETS = [
    ("Login Issues", "High", "I cannot log in to my account. Getting error 401."),
    ("Login Issues", "High", "Forgot my password and cannot reset it."),
    ("Login Issues", "High", "Account locked after too many failed attempts."),
    ("Login Issues", "Normal", "I keep getting logged out unexpectedly."),
    ("Login Issues", "Normal", "Login page is not loading on my phone."),
    ("Feature Request", "Normal", "Would be great to have dark mode in the app."),
    ("Feature Request", "Normal", "Please add multi-language support."),
    ("Feature Request", "Normal", "Can you add an export to PDF option?"),
    ("Feature Request", "Normal", "Add two-factor authentication for security."),
    ("Feature Request", "Normal", "Allow users to customize notifications."),
    ("Bug Report", "High", "App crashes when uploading large files."),
    ("Bug Report", "High", "The dashboard does not load on Safari browser."),
    ("Bug Report", "High", "Notifications are not being sent."),
    ("Bug Report", "High", "Search function returns incorrect results."),
    ("Bug Report", "Normal", "Minor UI glitch on settings page."),
    ("Bug Report", "Normal", "Profile picture upload fails sometimes."),
    ("Bug Report", "Normal", "App is slow after recent update."),
    ("Payment Issues", "High", "My payment did not go through but money was deducted."),
    ("Payment Issues", "High", "Unable to update my billing information."),
    ("Payment Issues", "High", "Received an invoice for the wrong amount."),
    ("Payment Issues", "High", "Subscription was cancelled without notice."),
    ("Payment Issues", "Normal", "How do I get a refund?"),
    ("Payment Issues", "Normal", "Can I change my payment method?"),
    ("Payment Issues", "Normal", "Is there a student discount?"),
    ("Account Management", "Normal", "How do I change my email address?"),
    ("Account Management", "Normal", "I want to delete my account permanently."),
    ("Account Management", "Normal", "How can I update my profile picture?"),
    ("Account Management", "Normal", "How do I change my username?"),
    ("Account Management", "Normal", "How do I enable two-factor authentication?"),
    ("Technical Support", "High", "The app is running very slowly."),
    ("Technical Support", "High", "Having trouble connecting to the server."),
    ("Technical Support", "High", "Getting a timeout error when saving changes."),
    ("Technical Support", "Normal", "App is not syncing across devices."),
    ("Technical Support", "Normal", "How do I clear the app cache?"),
    ("Technical Support", "Normal", "App is using too much battery."),
    ("Other", "Normal", "I have a question about your privacy policy."),
    ("Other", "Normal", "How do I contact customer support?"),
    ("Other", "Normal", "Where can I find the user manual?"),
    ("Other", "Normal", "Do you offer training sessions?"),
    ("Other", "Normal", "How do I unsubscribe from emails?"),
    ("Other", "Normal", "Can I suggest a new feature?"),
    ("Other", "Normal", "Is there a mobile version of the app?"),
    ("Other", "Normal", "How do I reset my preferences?"),
    ("Other", "Normal", "Can I use the app offline?"),
    ("Other", "Normal", "How do I report inappropriate content?"),
    ("Other", "Normal", "What is your data retention policy?"),
    ("Other", "Normal", "How do I update my contact information?"),
    ("Other", "Normal", "How do I access beta features?"),
]

@api.post("/mcp_generate")
async def mcp_generate(request: Request):
    import numpy as np
    import os
    from dotenv import load_dotenv
    print("[DEBUG] Loading .env file...")
    load_dotenv()
    print("[DEBUG] NEBIUS_API_KEY:", os.getenv("NEBIUS_API_KEY"))
    from llama_index.embeddings.nebius import NebiusEmbedding
    embed_model = NebiusEmbedding(api_key=os.getenv("NEBIUS_API_KEY"))
    EXAMPLE_EMBEDDINGS = [embed_model.get_query_embedding(ticket[2]) for ticket in EXAMPLE_TICKETS]
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    def classify_ticket(text):
        text_embedding = embed_model.get_query_embedding(text)
        similarities = [cosine_similarity(text_embedding, emb) for emb in EXAMPLE_EMBEDDINGS]
        top_indices = np.argsort(similarities)[-3:][::-1]
        print("\n[DEBUG] Top 3 matches for query:")
        for idx in top_indices:
            print(f"  Example: {EXAMPLE_TICKETS[idx][2]} | Category: {EXAMPLE_TICKETS[idx][0]} | Score: {similarities[idx]:.3f}")
        best_idx = int(np.argmax(similarities))
        best_similarity = similarities[best_idx]
        print(f"[DEBUG] Best match: {EXAMPLE_TICKETS[best_idx][2]} | Category: {EXAMPLE_TICKETS[best_idx][0]} | Score: {best_similarity:.3f}")
        if best_similarity < 0.5:
            return EXAMPLE_TICKETS[best_idx][0], "High"
        return EXAMPLE_TICKETS[best_idx][0], EXAMPLE_TICKETS[best_idx][1]
    data = await request.json()
    ticket_text = data.get("inputs", {}).get("ticket_text", "")
    if not ticket_text:
        raise HTTPException(status_code=400, detail="Missing ticket_text")
    category, priority = classify_ticket(ticket_text)
    reply = draft_reply.local(category, priority, ticket_text)
    return {"generated_text": {"category": category, "priority": priority, "reply": reply}}

# The rest of your FastAPI app and embedding logic should be under:
if __name__ == "__main__":
    import os
    import json
    import numpy as np
    from dotenv import load_dotenv
    from llama_index.embeddings.nebius import NebiusEmbedding
    from typing import List

    # Load environment variables
    load_dotenv()

    # Initialize Nebius embeddings
    embed_model = NebiusEmbedding(api_key=os.getenv("NEBIUS_API_KEY"))

    # 50 diverse example tickets for robust categorization
    EXAMPLE_TICKETS = [
        ("Login Issues", "High", "I cannot log in to my account. Getting error 401."),
        ("Login Issues", "High", "Forgot my password and cannot reset it."),
        ("Login Issues", "High", "Account locked after too many failed attempts."),
        ("Login Issues", "Normal", "I keep getting logged out unexpectedly."),
        ("Login Issues", "Normal", "Login page is not loading on my phone."),
        ("Feature Request", "Normal", "Would be great to have dark mode in the app."),
        ("Feature Request", "Normal", "Please add multi-language support."),
        ("Feature Request", "Normal", "Can you add an export to PDF option?"),
        ("Feature Request", "Normal", "Add two-factor authentication for security."),
        ("Feature Request", "Normal", "Allow users to customize notifications."),
        ("Bug Report", "High", "App crashes when uploading large files."),
        ("Bug Report", "High", "The dashboard does not load on Safari browser."),
        ("Bug Report", "High", "Notifications are not being sent."),
        ("Bug Report", "High", "Search function returns incorrect results."),
        ("Bug Report", "Normal", "Minor UI glitch on settings page."),
        ("Bug Report", "Normal", "Profile picture upload fails sometimes."),
        ("Bug Report", "Normal", "App is slow after recent update."),
        ("Payment Issues", "High", "My payment did not go through but money was deducted."),
        ("Payment Issues", "High", "Unable to update my billing information."),
        ("Payment Issues", "High", "Received an invoice for the wrong amount."),
        ("Payment Issues", "High", "Subscription was cancelled without notice."),
        ("Payment Issues", "Normal", "How do I get a refund?"),
        ("Payment Issues", "Normal", "Can I change my payment method?"),
        ("Payment Issues", "Normal", "Is there a student discount?"),
        ("Account Management", "Normal", "How do I change my email address?"),
        ("Account Management", "Normal", "I want to delete my account permanently."),
        ("Account Management", "Normal", "How can I update my profile picture?"),
        ("Account Management", "Normal", "How do I change my username?"),
        ("Account Management", "Normal", "How do I enable two-factor authentication?"),
        ("Technical Support", "High", "The app is running very slowly."),
        ("Technical Support", "High", "Having trouble connecting to the server."),
        ("Technical Support", "High", "Getting a timeout error when saving changes."),
        ("Technical Support", "Normal", "App is not syncing across devices."),
        ("Technical Support", "Normal", "How do I clear the app cache?"),
        ("Technical Support", "Normal", "App is using too much battery."),
        ("Other", "Normal", "I have a question about your privacy policy."),
        ("Other", "Normal", "How do I contact customer support?"),
        ("Other", "Normal", "Where can I find the user manual?"),
        ("Other", "Normal", "Do you offer training sessions?"),
        ("Other", "Normal", "How do I unsubscribe from emails?"),
        ("Other", "Normal", "Can I suggest a new feature?"),
        ("Other", "Normal", "Is there a mobile version of the app?"),
        ("Other", "Normal", "How do I reset my preferences?"),
        ("Other", "Normal", "Can I use the app offline?"),
        ("Other", "Normal", "How do I report inappropriate content?"),
        ("Other", "Normal", "What is your data retention policy?"),
        ("Other", "Normal", "How do I update my contact information?"),
        ("Other", "Normal", "How do I access beta features?"),
    ]

    # Precompute embeddings for example tickets
    EXAMPLE_EMBEDDINGS = [
        embed_model.get_query_embedding(ticket[2]) for ticket in EXAMPLE_TICKETS
    ]

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def classify_ticket(text):
        """Classify ticket using Nebius embeddings and cosine similarity."""
        try:
            text_embedding = embed_model.get_query_embedding(text)
            similarities = [cosine_similarity(text_embedding, emb) for emb in EXAMPLE_EMBEDDINGS]
            # Get top 3 matches for debugging
            top_indices = np.argsort(similarities)[-3:][::-1]
            print("\n[DEBUG] Top 3 matches for query:")
            for idx in top_indices:
                print(f"  Example: {EXAMPLE_TICKETS[idx][2]} | Category: {EXAMPLE_TICKETS[idx][0]} | Score: {similarities[idx]:.3f}")
            best_idx = int(np.argmax(similarities))
            best_similarity = similarities[best_idx]
            print(f"[DEBUG] Best match: {EXAMPLE_TICKETS[best_idx][2]} | Category: {EXAMPLE_TICKETS[best_idx][0]} | Score: {best_similarity:.3f}")
            if best_similarity < 0.5:
                return EXAMPLE_TICKETS[best_idx][0], "High"
            return EXAMPLE_TICKETS[best_idx][0], EXAMPLE_TICKETS[best_idx][1]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @api.post("/mcp_generate")
    async def mcp_generate(request: Request):
        """FastAPI endpoint for ticket processing."""
        try:
            data = await request.json()
            ticket_text = data.get("inputs", {}).get("ticket_text", "")
            if not ticket_text:
                raise HTTPException(status_code=400, detail="Missing ticket_text")
            
            # Classify ticket
            category, priority = classify_ticket(ticket_text)
            
            # Generate draft reply
            reply = draft_reply.local(category, priority, ticket_text)
            
            return {"generated_text": {"category": category, "priority": priority, "reply": reply}}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def process_ticket(ticket_text):
        """Gradio interface function for processing tickets (direct call, no HTTP)."""
        try:
            # Directly classify ticket
            category, priority = classify_ticket(ticket_text)
            # Directly generate draft reply
            reply = draft_reply.local(category, priority, ticket_text)
            return category, priority, reply
        except Exception as e:
            return str(e), "Error", "Error occurred while processing ticket"

    # Optionally, add code to run FastAPI locally if needed
