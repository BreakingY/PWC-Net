# Copyright (C) 2026 sunkx
# Licensed under the GNU General Public License v3.0
import cv2
import numpy as np
import onnxruntime as ort


def create_session(onnx_path):
    return ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )


def preprocess(img, target_h=448, target_w=1024):
    """
    BGR -> CHW -> float32
    """
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    # img = img[:, :, ::-1]  # BGR -> RGB

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW

    return img


def postprocess_flow(flow, orig_h, orig_w, net_h=448, net_w=1024):
    """
    flow: [1,2,H,W]
    return: [2,orig_h,orig_w]
    """

    flow = flow[0]  # [2,H,W]

    flow_resized = np.stack([
        cv2.resize(flow[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR),
        cv2.resize(flow[1], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    ], axis=0)

    # scale displacement back to original resolution
    flow_resized[0] *= orig_w / net_w
    flow_resized[1] *= orig_h / net_h

    return flow_resized


def infer_batch1(session, img1, img2):
    """
    return: HWC flow
    """

    h, w = img1.shape[:2]

    one = preprocess(img1)
    two = preprocess(img2)

    one = np.expand_dims(one, 0)  # [1,3,448,1024]
    two = np.expand_dims(two, 0)

    flow = session.run(
        None,
        {
            "input1": one,
            "input2": two
        }
    )[0]  # [1,2,448,1024]

    flow = postprocess_flow(flow, h, w)

    return np.transpose(flow, (1, 2, 0))  # HWC


def infer_batch2(session, img1, img2):
    """
    batch=2 inference (PWC-style correct pairing)

    input:
        img1, img2: BGR images (H,W,3)

    return:
        [flow1, flow2] each HWC
    """

    h, w = img1.shape[:2]

    img1 = preprocess(img1)
    img2 = preprocess(img2)

    batch_one = np.stack([img1, img1], axis=0)  # [2,3,448,1024]
    batch_two = np.stack([img2, img2], axis=0)  # [2,3,448,1024]

    flows = session.run(
        None,
        {
            "input1": batch_one,
            "input2": batch_two
        }
    )[0]  # [2,2,448,1024]

    results = []

    for i in range(2):
        flow = flows[i:i+1]  # [1,2,H,W]
        flow = postprocess_flow(flow, h, w)
        flow = np.transpose(flow, (1, 2, 0))  # HWC
        results.append(flow)

    return results


def flow_to_hsv(flow):

    if flow.shape[-1] == 2:
        fx = flow[..., 0]
        fy = flow[..., 1]
    else:
        fx = flow[0]
        fy = flow[1]

    mag, ang = cv2.cartToPolar(fx, fy)

    h, w = fx.shape

    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    hue = np.zeros_like(ang, dtype=np.float32)

    ang = np.mod(ang, 2 * np.pi)

    mask1 = (ang < np.pi / 2)
    hue[mask1] = (ang[mask1] / (np.pi / 2)) * 30

    mask2 = (ang >= np.pi / 2) & (ang < np.pi)
    hue[mask2] = 30 + ((ang[mask2] - np.pi / 2) / (np.pi / 2)) * 90

    mask3 = (ang >= np.pi)
    hue[mask3] = 120 + ((ang[mask3] - np.pi) / np.pi) * 60

    hsv[..., 0] = np.clip(hue, 0, 179).astype(np.uint8)

    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 1] = mag_norm.astype(np.uint8)

    hsv[..., 2] = 255

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb
if __name__ == "__main__":

    onnx_path = "pwcnet.onnx"
    session = create_session(onnx_path)

    img1 = cv2.imread("./images/one.png")
    img2 = cv2.imread("./images/two.png")


    flow1 = infer_batch1(session, img1, img2)
    print("Batch1 flow shape:", flow1.shape)


    flows = infer_batch2(session, img1, img2)
    print("Batch2 flow shapes:", [f.shape for f in flows])

    vis = flow_to_hsv(flows[1])
    cv2.imwrite("flow_onnx.jpg", vis)