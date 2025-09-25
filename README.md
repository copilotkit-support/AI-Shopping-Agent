# AI Shopping Demo

A demo project with a Next.js frontend and FastAPI backend Powered by Tavily and CopilotKit.

![Demo](assets/example.gif)

---

## Project Structure

- `frontend/` — Next.js 15 app (UI)
- `agent/` — FastAPI backend (Agent)

---

## Getting Started

### 1. Environment Configuration

Create a `.env` file in each relevant directory as needed. 

#### Frontend (`frontend/.env`):
```env
OPENAI_API_KEY=<<your-openai-key-here>>
```
#### agent (`agent/.env`):
```env
OPENAI_API_KEY=<<your-openai-key-here>>
TAVILY_API_KEY = <<your-tavily-key-here>>
```


---


### 2. Run the Backend agent

```bash
cd agent
poetry install
poetry run python main.py
```

---


### 3. Start the Frontend

```bash
cd frontend
pnpm install
pnpm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to view the app.

---

## Notes
- Update environment variables as needed for your deployment.

---

### Hosted URL : https://ai-shopping-assistant-xi.vercel.app/