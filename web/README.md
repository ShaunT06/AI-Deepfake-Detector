# DeepScan — Web Frontend

Next.js (App Router) frontend for the [DeepScan deepfake detector](https://github.com/ShaunT06/AI-Deepfake-Detector).
Deployed to Vercel; calls the FastAPI inference backend in `../backend`
over HTTP (no model inference happens in this app).

## Local development

```bash
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL to your backend's URL
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Environment variables

- `NEXT_PUBLIC_API_URL` — base URL of the deployed inference API (see
  `../backend`), no trailing slash. Required — without it, the app shows
  a config error instead of trying to call an undefined endpoint.

## Deploy

```bash
vercel --prod
```

or connect this repo to Vercel and set the **Root Directory** to `web/`
in the project settings, with `NEXT_PUBLIC_API_URL` set as an
environment variable there too.
