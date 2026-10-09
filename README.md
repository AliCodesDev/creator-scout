# Creator Scout

Takes a campaign brief in plain English (e.g. *"Arabic-speaking beauty creators in Riyadh, 50k–300k followers, natural hair content"*) and returns a ranked shortlist of creators, each with a fit score, reasons that cite specific posts, and the cost of the search.

The LLM parses the brief and explains the results. Filtering and ranking are deterministic code.

## Run locally

```bash
cp .env.example .env   # add your LLM API key
make db                # Postgres + pgvector on localhost:5434
make help              # list all commands
```

*Work in progress.*
