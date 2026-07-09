from dotenv import load_dotenv
load_dotenv()

from core.graph import agent_app

try:
    png_bytes = agent_app.get_graph().draw_mermaid_png()
    with open("new_graph.png", "wb") as f:
        f.write(png_bytes)
    print("Successfully saved new_graph.png")
except Exception as e:
    print(f"Error drawing graph: {e}")
