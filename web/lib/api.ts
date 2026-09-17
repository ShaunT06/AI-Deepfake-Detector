export interface PredictResponse {
  is_fake: boolean;
  fake_prob: number;
  real_prob: number;
  backbone: string;
  labels_verified: boolean;
  img_size: number;
  gradcam_image_base64: string;
}

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export class ApiError extends Error {}

export async function predict(file: File): Promise<PredictResponse> {
  if (!API_URL) {
    throw new ApiError(
      "NEXT_PUBLIC_API_URL isn't configured. Set it to your deployed inference API's URL.",
    );
  }

  const form = new FormData();
  form.append("file", file);

  let res: Response;
  try {
    res = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
  } catch {
    throw new ApiError(
      "Couldn't reach the inference API. It may be waking up from a cold start (this can take up to a minute on a free tier) — please try again shortly.",
    );
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON; fall back to statusText
    }
    throw new ApiError(detail || `Request failed with status ${res.status}`);
  }

  return res.json();
}
