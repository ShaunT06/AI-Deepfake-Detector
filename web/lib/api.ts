export interface PredictResponse {
  is_fake: boolean;
  fake_prob: number;
  real_prob: number;
  backbone: string;
  labels_verified: boolean;
  img_size: number;
  gradcam_image_base64: string;
}

// Inference runs inside this same Vercel deployment (web/api/predict.py,
// onnxruntime-based — see scripts/export_onnx.py) — no external service,
// no NEXT_PUBLIC_API_URL, no CORS.
export const MAX_UPLOAD_BYTES = 3.5 * 1024 * 1024; // matches web/api/predict.py's own limit,
// set below Vercel's ~4.5MB request body cap.

export class ApiError extends Error {}

export async function predict(file: File): Promise<PredictResponse> {
  let res: Response;
  try {
    res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": file.type },
      body: file,
    });
  } catch {
    throw new ApiError("Couldn't reach the inference API — please try again shortly.");
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
