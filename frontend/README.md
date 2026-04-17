# HyperNews Frontend

This is the Next.js frontend for HyperNews.

## Run Locally

```bash
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open:

```text
http://localhost:3000
```

## Backend Requirement

The frontend expects the FastAPI backend to be running on:

```text
http://localhost:8000
```

You can override that with `NEXT_PUBLIC_API_URL`.

## Notes

- `next.config.ts` pins Turbopack's root to the frontend directory to avoid workspace-root detection issues.
- If Turbopack becomes too heavy on your machine, you can use Webpack instead:

```bash
npx next dev --webpack
```

- The UI talks to:
  - `/recommend`
  - `/feedback`
  - `/graph`
  - `/profile/:user_id`
  - `/reset/:user_id`
