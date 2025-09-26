from fastapi import FastAPI
import uvicorn
# from copilotkit
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint
from copilotkit.crewai.crewai_agent import CrewAIAgent
import os
app = FastAPI()
from shopping_assistant import graph
from shopping import ShoppingAgentFlow
from shopping import AgentState


sdk = CopilotKitRemoteEndpoint(
    agents=[
        # LangGraphAgent(
        #     name="shopping_agent",
        #     description="A shopping agent that can help you find the best products for your needs by searching various retailers",
        #     graph=graph
        # ),
        CrewAIAgent(
            name="shopping_agent_crewai",
            description="A shopping agent that can help you find the best products for your needs by searching various retailers",
            flow= ShoppingAgentFlow()
        )
    ]
)

add_fastapi_endpoint(app, sdk, "/copilotkit")


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        timeout_keep_alive=900,  # 15 minutes = 900 seconds
        timeout_graceful_shutdown=900,  # 15 minutes graceful shutdown
        reload_dirs=(
            ["."] +
            (["../../../sdk-python/copilotkit"]
             if os.path.exists("../../../sdk-python/copilotkit")
             else []
             )
        )
    )
    
if __name__ == "__main__":
    main()