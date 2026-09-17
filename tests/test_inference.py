from PIL import Image

from deepscan.inference import Predictor


def test_predict_end_to_end(tiny_checkpoint):
    predictor = Predictor(tiny_checkpoint)
    image = Image.new("RGB", (300, 300), color=(120, 160, 200))

    pred = predictor.predict(image)

    assert isinstance(pred.is_fake, bool)
    assert 0.0 <= pred.fake_prob <= 1.0
    assert 0.0 <= pred.real_prob <= 1.0
    assert abs(pred.fake_prob + pred.real_prob - 1.0) < 1e-4
    assert pred.labels_verified is True  # fixture embeds class_to_idx
    assert pred.gradcam_image.size == (predictor.img_size, predictor.img_size)
