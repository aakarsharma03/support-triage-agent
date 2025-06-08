import gradio as gr
import requests

def process_ticket(ticket_text):
    try:
        response = requests.post(
            "http://localhost:8000/mcp_generate",
            json={"inputs": {"ticket_text": ticket_text}}
        )
        response.raise_for_status()
        result = response.json()["generated_text"]
        return result["category"], result["priority"], result["reply"]
    except Exception as e:
        return str(e), "Error", "Error occurred while processing ticket"

with gr.Blocks() as demo:
    gr.Markdown("# Support Ticket Triage & Draft Response")
    with gr.Row():
        ticket_input = gr.Textbox(
            label="Describe your issue",
            placeholder="Please describe your issue in detail...",
            lines=5
        )
    submit_btn = gr.Button("Submit Ticket")
    with gr.Row():
        category_output = gr.Textbox(label="Category", interactive=False)
        priority_output = gr.Textbox(label="Priority", interactive=False)
    reply_output = gr.Textbox(
        label="Draft Reply",
        interactive=False,
        lines=5
    )
    submit_btn.click(
        fn=process_ticket,
        inputs=ticket_input,
        outputs=[category_output, priority_output, reply_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860) 